import numpy as np

class ReceiverObservation:
    def __init__(
        self,
        tuned_band,
        dwell_start_time,
        dwell_end_time,
        observation_time,
        effective_duration,
        detection_result,
        switching_time=0,
        settling_time=0,
        snr_db=None,
        received_power_dbm=None,
        effective_pd=None,
        effective_pfa=None
    ):
        self.tuned_band = tuned_band
        self.dwell_start_time = dwell_start_time
        self.dwell_end_time = dwell_end_time
        self.observation_time = observation_time
        self.effective_duration = effective_duration
        self.detection_result = detection_result
        self.switching_time = switching_time
        self.settling_time = settling_time
        self.snr_db = snr_db
        self.received_power_dbm = received_power_dbm
        self.effective_pd = effective_pd
        self.effective_pfa = effective_pfa

class VirtualReceiver:
    """
    Realistic EW Spectrum Monitoring Receiver Model.

    Physical & Operational Assumptions:
    1. RF Noise Floor: Modeled as thermal noise k*T*B (default approx -100 dBm for configured instantaneous BW).
    2. Received Power & SNR:
       - Emitter transmit power P_tx (default ~ 20 dBm) attenuated by distance/path loss:
         P_rx = P_tx - path_loss_db
       - SNR (dB) = P_rx (dBm) - Noise_Floor (dBm).
    3. Detection Probability P_d(SNR, dwell):
       - Modulated by a standard EW logistic / Marcum-Q approximation:
         P_d_base = 1.0 / (1.0 + exp(-steepness * (SNR - detection_threshold_snr)))
       - Dwell Integration: Longer dwell increases effective non-coherent integration:
         P_d_eff = 1.0 - (1.0 - P_d_base) ** dwell_time
    4. False Alarm Probability P_fa:
       - Constant false alarm rate (CFAR) or thermal noise exceedance:
         P_fa_eff = 1.0 - (1.0 - false_alarm_probability) ** dwell_time
    5. Tuner Retuning & Settling:
       - Retuning between bands consumes `switching_time` time steps.
       - Optional synthesizer `settling_time` adds dead time during retuning.
    6. Instantaneous Bandwidth (IBW):
       - Default IBW covers 1 channel/band cleanly; out-of-band rejection prevents spurious detection.
    """

    def __init__(
        self,
        num_bands=10,
        detection_probability=0.90,
        false_alarm_probability=0.05,
        switching_time=0,
        settling_time=0,
        noise_floor_dbm=-100.0,
        detection_threshold_snr_db=10.0,
        instantaneous_bandwidth_mhz=20.0,
        power_model_enabled=False,
        receiver_position=(0.0, 0.0),
        snr_steepness=0.35,
        seed=None
    ):
        self.num_bands = num_bands
        self.detection_probability = detection_probability
        self.false_alarm_probability = false_alarm_probability
        self.switching_time = switching_time
        self.settling_time = settling_time
        self.noise_floor_dbm = noise_floor_dbm
        self.detection_threshold_snr_db = detection_threshold_snr_db
        self.instantaneous_bandwidth_mhz = instantaneous_bandwidth_mhz
        self.power_model_enabled = power_model_enabled
        self.receiver_position = receiver_position
        self.snr_steepness = snr_steepness
        self.seed = seed

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()
            
        self.current_band = None
        self.scan_history = []
        # Evaluator only history
        self.evaluator_truth_history = []

    def reset(self, seed=None):
        """
        Reset scan history and optionally reseed the detector RNG.
        """
        self.scan_history = []
        self.evaluator_truth_history = []
        self.current_band = None
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def calculate_snr(self, emitter, band: int) -> float:
        """
        Calculate received SNR in dB for an emitter on the tuned band.
        Assumes log-distance path loss if coordinates exist, or default nominal power.
        """
        if not self.power_model_enabled:
            return self.detection_threshold_snr_db + 15.0  # high nominal SNR for classic mode

        p_tx = getattr(emitter, 'power_dbm', 20.0)
        em_pos = getattr(emitter, 'position', (0.0, 10.0))
        dist = np.sqrt((self.receiver_position[0] - em_pos[0])**2 + (self.receiver_position[1] - em_pos[1])**2)
        dist = max(1.0, float(dist))

        # Free space path loss approximation: PL = 20*log10(dist) + 32.44 + 20*log10(freq_mhz)
        # Using normalized path loss: 20 * log10(dist) + 20.0
        path_loss_db = 20.0 * np.log10(dist) + 20.0
        p_rx_dbm = p_tx - path_loss_db
        snr_db = p_rx_dbm - self.noise_floor_dbm
        return float(snr_db)

    def compute_detection_probabilities(self, signal_present: bool, snr_db: float, dwell_time: int):
        """
        Compute effective Pd and Pfa under SNR and dwell time.
        """
        if not self.power_model_enabled:
            # Baseline compatibility mode: use configured detection_probability and false_alarm_probability
            pd_eff = 1.0 - (1.0 - self.detection_probability) ** dwell_time
            pfa_eff = 1.0 - (1.0 - self.false_alarm_probability) ** dwell_time
            return pd_eff, pfa_eff

        if signal_present:
            # Logistic curve around threshold SNR
            # High SNR -> Pd -> 1.0; Low SNR -> Pd -> 0.0
            pd_single = 1.0 / (1.0 + np.exp(-self.snr_steepness * (snr_db - self.detection_threshold_snr_db)))
            # Dwell integration (independent samples per dwell unit)
            pd_eff = 1.0 - (1.0 - pd_single) ** dwell_time
        else:
            pd_eff = 0.0

        # Pfa thermal noise false alarm scaling with dwell
        pfa_eff = 1.0 - (1.0 - self.false_alarm_probability) ** dwell_time
        return float(pd_eff), float(pfa_eff)

    def scan(self, environment, band, dwell_time=1):
        # Calculate switching and settling time
        actual_switching_time = 0
        actual_settling_time = 0
        if self.current_band is not None and self.current_band != band:
            actual_switching_time = self.switching_time
            actual_settling_time = self.settling_time
            
        self.current_band = band
        total_retune_delay = actual_switching_time + actual_settling_time
        
        # Advance environment for retuning delay
        for _ in range(total_retune_delay):
            environment.step()

        dwell_start_time = environment.time
        detections = []
        any_signal_present = False
        step_snr_db = None
        step_prx_dbm = None

        for _ in range(dwell_time):
            active_bands = environment.step()
            signal_present = band in active_bands
            if signal_present:
                any_signal_present = True

            # If signal present and emitters are inspectable, compute active SNR
            active_emitter = None
            if signal_present and hasattr(environment, 'emitters'):
                for em in environment.emitters:
                    if getattr(em, 'band', None) == band and getattr(em, 'active', False):
                        active_emitter = em
                        break

            snr_db = self.calculate_snr(active_emitter, band) if signal_present else -20.0
            step_snr_db = snr_db
            step_prx_dbm = snr_db + self.noise_floor_dbm

            pd_eff, pfa_eff = self.compute_detection_probabilities(signal_present, snr_db, dwell_time=1)

            rand_val = self.rng.random()
            if signal_present:
                detected = (rand_val < pd_eff)
            else:
                detected = (rand_val < pfa_eff)

            detections.append(detected)

            # Store truth separately for evaluators
            self.evaluator_truth_history.append({
                "time": environment.time,
                "band": band,
                "signal_present": signal_present
            })

        dwell_end_time = environment.time
        final_detection = any(detections)

        # Compute summary Pd/Pfa over the full dwell for observation telemetry
        summary_pd, summary_pfa = self.compute_detection_probabilities(
            any_signal_present,
            step_snr_db if step_snr_db is not None else -20.0,
            dwell_time
        )
        
        obs = ReceiverObservation(
            tuned_band=band,
            dwell_start_time=dwell_start_time,
            dwell_end_time=dwell_end_time,
            observation_time=dwell_end_time,
            effective_duration=dwell_time,
            detection_result=final_detection,
            switching_time=actual_switching_time,
            settling_time=actual_settling_time,
            snr_db=step_snr_db,
            received_power_dbm=step_prx_dbm,
            effective_pd=summary_pd,
            effective_pfa=summary_pfa
        )
        
        # Store for observable history (Option B semantic contract)
        self.scan_history.append({
            "dwell_start_time": obs.dwell_start_time,
            "dwell_end_time": obs.dwell_end_time,
            "observation_time": obs.observation_time,
            "effective_duration": obs.effective_duration,
            "tuned_band": obs.tuned_band,
            "detection_result": obs.detection_result,
            "switching_time": obs.switching_time,
            "settling_time": obs.settling_time,
            "snr_db": obs.snr_db,
            "received_power_dbm": obs.received_power_dbm,
            "effective_pd": obs.effective_pd,
            "effective_pfa": obs.effective_pfa,
            # Backward compat for legacy metrics
            "time": obs.observation_time,
            "band": obs.tuned_band,
            "detected": obs.detection_result,
            "signal_present": any_signal_present 
        })

        # Return boolean for backwards compatibility
        return final_detection

    def get_last_result(self):
        if not self.scan_history:
            return None
        return self.scan_history[-1]