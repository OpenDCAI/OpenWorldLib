from ..communicator.communicator import Communicator
from ..utils.logger import Logger


class WeatherManager:
    """Weather manager that controls the weather on the fly."""

    def __init__(self, communicator: Communicator):
        self.communicator = communicator
        self.logger = Logger.get_logger('WeatherManager')
        self.name = 'GEN_WeatherManager'

        self.sun_azimuth_angle = 0      # pitch
        self.sun_altitude_angle = 0     # yaw
        self.sun_intensity = 0

        self.fog_density = 0            # 0-100
        self.fog_distance = 0           # 0-5000
        self.fog_falloff = 0            # 0-2

        self.rayleigh_scattering_scale = 0
        self.mie_scattering_scale = 0

    ##############################################################
    # Sun-related methods
    ##############################################################

    def get_sun_direction(self):
        """Get current sun direction.
        
        Returns:
            tuple: (pitch, yaw) in degrees
        """
        try:
            # Get from UnrealCV through communicator
            pitch, yaw = self.communicator.get_sun_direction(self.name)
            if pitch is not None and yaw is not None:
                # Update local state with actual values from UnrealCV
                self.sun_azimuth_angle = pitch
                self.sun_altitude_angle = yaw
                return pitch, yaw
            else:
                # Return local state if UnrealCV call failed
                return self.sun_azimuth_angle, self.sun_altitude_angle
        except Exception as e:
            self.logger.warning(f"Failed to get sun direction from UnrealCV: {e}")
            return self.sun_azimuth_angle, self.sun_altitude_angle
        
    def get_sun_intensity(self):    
        """Get current sun intensity.
        
        Returns:
            Sun intensity.
        """
        try:
            intensity = self.communicator.get_sun_intensity(self.name)
            if intensity is not None:
                self.sun_intensity = intensity
                return intensity
            else:
                return self.sun_intensity
        except Exception as e:
            self.logger.warning(f"Failed to get sun intensity from UnrealCV: {e}")
            return self.sun_intensity
    
    def set_sun_direction(self, pitch: float, yaw: float):
        """Set sun direction.
        
        Args:
            pitch: Pitch of the sun. Range: -89 to 89 degrees
            yaw: Yaw of the sun. Range: any value (will be normalized to 0-360)
        """
        # Handle pitch (elevation angle) - clamp to valid range
        if not -89 <= pitch <= 89:
            self.logger.warning(f"Sun pitch {pitch} is out of range [-89, 89]. Clamping to valid range.")
            pitch = max(-89, min(89, pitch))
        
        # Handle yaw (azimuth angle) - normalize to 0-360 range
        normalized_yaw = self._normalize_angle(yaw)
        if normalized_yaw != yaw:
            self.logger.info(f"Sun yaw {yaw} normalized to {normalized_yaw} degrees")
        
        self.sun_azimuth_angle = pitch
        self.sun_altitude_angle = normalized_yaw
        
        # Call UnrealCV method through communicator
        self.communicator.set_sun_direction(self.name, pitch, normalized_yaw)
        self.logger.info(f"Sun direction set to pitch={pitch}, yaw={normalized_yaw}")

    def set_sun_azimuth_angle(self, pitch: float):
        """Set sun azimuth angle (pitch) only.
        
        Args:
            pitch: Pitch of the sun. Range: -89 to 89 degrees
        """
        current_yaw = self.sun_altitude_angle
        self.set_sun_direction(pitch, current_yaw)

    def set_sun_altitude_angle(self, yaw: float):
        """Set sun altitude angle (yaw) only.
        
        Args:
            yaw: Yaw of the sun. Any value (will be normalized to 0-360)
        """
        current_pitch = self.sun_azimuth_angle
        self.set_sun_direction(current_pitch, yaw)

    def set_sun_intensity(self, intensity: float):
        """Set sun intensity.
        
        Args:
            intensity: Sun intensity value
        """
        self.sun_intensity = intensity
        self.communicator.set_sun_intensity(self.name, intensity)
        self.logger.info(f"Sun intensity set to {intensity}")

    ##############################################################
    # Fog-related methods
    ##############################################################

    def get_fog(self):
        """Get current fog parameters.
        
        Returns:
            tuple: (density, distance, falloff)
        """
        try:
            # Get from UnrealCV through communicator
            density, distance, falloff = self.communicator.get_fog(self.name)
            if density is not None and distance is not None and falloff is not None:
                # Update local state with actual values from UnrealCV
                self.fog_density = density
                self.fog_distance = distance
                self.fog_falloff = falloff
                return density, distance, falloff
            else:
                # Return local state if UnrealCV call failed
                return self.fog_density, self.fog_distance, self.fog_falloff
        except Exception as e:
            self.logger.warning(f"Failed to get fog from UnrealCV: {e}")
            return self.fog_density, self.fog_distance, self.fog_falloff
    
    def set_fog(self, density: float, distance: float, falloff: float):
        """Set fog parameters.
        
        Args:
            density: Fog density. Range: 0-100
            distance: Fog distance in cm. Range: 0-5000
            falloff: Fog falloff. Range: 0-2
        """
        # Validate and clamp values to valid ranges
        if not 0 <= density <= 100:
            self.logger.warning(f"Fog density {density} is out of range [0, 100]. Clamping to valid range.")
            density = max(0, min(100, density))
        
        if not 0 <= distance <= 5000:
            self.logger.warning(f"Fog distance {distance} is out of range [0, 5000]. Clamping to valid range.")
            distance = max(0, min(5000, distance))
        
        if not 0 <= falloff <= 2:
            self.logger.warning(f"Fog falloff {falloff} is out of range [0, 2]. Clamping to valid range.")
            falloff = max(0, min(2, falloff))
        
        self.fog_density = density
        self.fog_distance = distance
        self.fog_falloff = falloff
        
        # Call UnrealCV method through communicator
        self.communicator.set_fog(self.name, density, distance, falloff)
        self.logger.info(f"Fog set to density={density}, distance={distance}, falloff={falloff}")

    def set_fog_density(self, density: float):
        """Set fog density only.
        
        Args:
            density: Fog density. Range: 0-100
        """
        current_distance, current_falloff = self.fog_distance, self.fog_falloff
        self.set_fog(density, current_distance, current_falloff)

    def set_fog_distance(self, distance: float):
        """Set fog distance only.
        
        Args:
            distance: Fog distance in cm. Range: 0-5000
        """
        current_density, current_falloff = self.fog_density, self.fog_falloff
        self.set_fog(current_density, distance, current_falloff)

    def set_fog_falloff(self, falloff: float):
        """Set fog falloff only.
        
        Args:
            falloff: Fog falloff. Range: 0-2
        """
        current_density, current_distance = self.fog_density, self.fog_distance
        self.set_fog(current_density, current_distance, falloff)

    ##############################################################
    # Atmosphere-related methods
    ##############################################################

    def get_atmosphere(self):
        """Get current atmosphere parameters.
        
        Returns:
            tuple: (rayleigh_scattering_scale, mie_scattering_scale)
        """
        try:
            # Get from UnrealCV through communicator
            rayleigh, mie = self.communicator.get_atmosphere(self.name)
            if rayleigh is not None and mie is not None:
                # Update local state with actual values from UnrealCV
                self.rayleigh_scattering_scale = rayleigh
                self.mie_scattering_scale = mie
                return rayleigh, mie
            else:
                # Return local state if UnrealCV call failed
                return self.rayleigh_scattering_scale, self.mie_scattering_scale
        except Exception as e:
            self.logger.warning(f"Failed to get atmosphere from UnrealCV: {e}")
            return self.rayleigh_scattering_scale, self.mie_scattering_scale
    
    def set_atmosphere(self, rayleigh: float, mie: float):
        """Set atmosphere parameters.
        
        Args:
            rayleigh: Rayleigh scattering scale. Range: 0-2
            mie: Mie scattering scale. Range: 0-5
        """
        # Validate and clamp values to valid ranges
        if not 0 <= rayleigh <= 2:
            self.logger.warning(f"Rayleigh scattering scale {rayleigh} is out of range [0, 2]. Clamping to valid range.")
            rayleigh = max(0, min(2, rayleigh))
        
        if not 0 <= mie <= 5:
            self.logger.warning(f"Mie scattering scale {mie} is out of range [0, 5]. Clamping to valid range.")
            mie = max(0, min(5, mie))
        
        self.rayleigh_scattering_scale = rayleigh
        self.mie_scattering_scale = mie
        
        # Call UnrealCV method through communicator
        self.communicator.set_atmosphere(self.name, rayleigh, mie)
        self.logger.info(f"Atmosphere set to rayleigh={rayleigh}, mie={mie}")

    def set_rayleigh_scattering_scale(self, rayleigh: float):
        """Set Rayleigh scattering scale only.
        
        Args:
            rayleigh: Rayleigh scattering scale. Range: 0-2
        """
        current_mie = self.mie_scattering_scale
        self.set_atmosphere(rayleigh, current_mie)

    def set_mie_scattering_scale(self, mie: float):
        """Set Mie scattering scale only.
        
        Args:
            mie: Mie scattering scale. Range: 0-5
        """
        current_rayleigh = self.rayleigh_scattering_scale
        self.set_atmosphere(current_rayleigh, mie)

    ##############################################################
    # Preset weather methods
    ##############################################################
    
    # def set_sunny_weather(self):
    #     """Set sunny weather preset."""
    #     self.set_sun_direction(45, 180)  # Midday sun
    #     self.set_sun_intensity(1.0)
    #     self.set_fog(0, 0, 0)  # No fog
    #     self.set_atmosphere(0.1, 0.1)  # Clear atmosphere
    #     self.logger.info("Weather set to sunny preset")

    # def set_cloudy_weather(self):
    #     """Set cloudy weather preset."""
    #     self.set_sun_direction(30, 200)  # Lower sun angle
    #     self.set_sun_intensity(0.6)
    #     self.set_fog(10, 2000, 0.5)  # Light fog
    #     self.set_atmosphere(0.3, 0.2)  # Slightly hazy atmosphere
    #     self.logger.info("Weather set to cloudy preset")

    # def set_foggy_weather(self):
    #     """Set foggy weather preset."""
    #     self.set_sun_direction(15, 220)  # Low sun angle
    #     self.set_sun_intensity(0.4)
    #     self.set_fog(70, 500, 1.5)  # Heavy fog
    #     self.set_atmosphere(0.5, 0.4)  # Hazy atmosphere
    #     self.logger.info("Weather set to foggy preset")

    # def set_sunset_weather(self):
    #     """Set sunset weather preset."""
    #     self.set_sun_direction(5, 270)  # Very low sun angle
    #     self.set_sun_intensity(0.8)
    #     self.set_fog(20, 1500, 0.8)  # Light fog
    #     self.set_atmosphere(0.8, 0.6)  # Warm atmosphere
    #     self.logger.info("Weather set to sunset preset")

    # def set_night_weather(self):
    #     """Set night weather preset."""
    #     self.set_sun_direction(-10, 0)  # Sun below horizon
    #     self.set_sun_intensity(0.1)
    #     self.set_fog(5, 3000, 0.3)  # Very light fog
    #     self.set_atmosphere(0.2, 0.1)  # Clear night atmosphere
    #     self.logger.info("Weather set to night preset")

    ##############################################################
    # Utility methods
    ##############################################################
    
    def set_weather_from_dict(self, weather_config: dict):
        """Set weather from a configuration dictionary.
        
        Args:
            weather_config: Dictionary containing weather parameters
                Example:
                {
                    'sun_direction': {'pitch': 45, 'yaw': 180},
                    'sun_intensity': 1.0,
                    'fog': {'density': 0, 'distance': 0, 'falloff': 0},
                    'atmosphere': {'rayleigh': 0.1, 'mie': 0.1}
                }
        """
        try:
            # Set sun direction
            if 'sun_direction' in weather_config:
                sun_dir = weather_config['sun_direction']
                pitch = sun_dir.get('pitch', self.sun_azimuth_angle)
                yaw = sun_dir.get('yaw', self.sun_altitude_angle)
                self.set_sun_direction(pitch, yaw)
            
            # Set sun intensity
            if 'sun_intensity' in weather_config:
                self.set_sun_intensity(weather_config['sun_intensity'])
            
            # Set fog
            if 'fog' in weather_config:
                fog = weather_config['fog']
                density = fog.get('density', self.fog_density)
                distance = fog.get('distance', self.fog_distance)
                falloff = fog.get('falloff', self.fog_falloff)
                self.set_fog(density, distance, falloff)
            
            # Set atmosphere
            if 'atmosphere' in weather_config:
                atmosphere = weather_config['atmosphere']
                rayleigh = atmosphere.get('rayleigh', self.rayleigh_scattering_scale)
                mie = atmosphere.get('mie', self.mie_scattering_scale)
                self.set_atmosphere(rayleigh, mie)
            
            self.logger.info("Weather set from configuration dictionary")
        except Exception as e:
            self.logger.error(f"Failed to set weather from configuration: {e}")

    def sync_with_unreal(self):
        """Synchronize local weather state with Unreal Engine.
        
        Args:
            weather_manager_name: Name of the weather manager in Unreal Engine.
        """
        try:
            # Get all weather info from UnrealCV
            weather_info = self.communicator.get_weather_info(self.name)
            if weather_info:
                # Update sun direction
                if 'sun_direction' in weather_info:
                    sun_dir = weather_info['sun_direction']
                    self.sun_azimuth_angle = sun_dir['pitch']
                    self.sun_altitude_angle = sun_dir['yaw']
                
                # Update fog parameters
                if 'fog' in weather_info:
                    fog = weather_info['fog']
                    self.fog_density = fog['density']
                    self.fog_distance = fog['distance']
                    self.fog_falloff = fog['falloff']
                
                # Update atmosphere parameters
                if 'atmosphere' in weather_info:
                    atmosphere = weather_info['atmosphere']
                    self.rayleigh_scattering_scale = atmosphere['rayleigh']
                    self.mie_scattering_scale = atmosphere['mie']
                
                self.logger.info("Successfully synchronized weather state with Unreal Engine")
            else:
                self.logger.warning("Failed to get weather info from Unreal Engine")
        except Exception as e:
            self.logger.error(f"Failed to sync with Unreal Engine: {e}")

    
    def spawn(self, model_path = '/Game/Weather/BP_WeatherManager.BP_WeatherManager_C'):
        """Spawn weather manager.
        
        Args:
            model_path: Path to the weather manager model.
        """
        self.communicator.spawn_weather_manager(self.name, model_path)

    # def set_weather_from_json(self, json_file_path: str):
    #     """Set weather from a JSON configuration file.
        
    #     Args:
    #         json_file_path: Path to the JSON configuration file
    #     """
    #     try:
    #         import json
    #         with open(json_file_path, 'r') as f:
    #             weather_config = json.load(f)
    #         self.set_weather_from_dict(weather_config)
    #     except Exception as e:
    #         self.logger.error(f"Failed to load weather configuration from {json_file_path}: {e}")

    # def set_weather_from_preset(self, preset_name: str):
    #     """Set weather from a named preset.
        
    #     Args:
    #         preset_name: Name of the preset ('sunny', 'cloudy', 'foggy', 'sunset', 'night')
    #     """
    #     preset_methods = {
    #         'sunny': self.set_sunny_weather,
    #         'cloudy': self.set_cloudy_weather,
    #         'foggy': self.set_foggy_weather,
    #         'sunset': self.set_sunset_weather,
    #         'night': self.set_night_weather
    #     }
        
    #     if preset_name in preset_methods:
    #         preset_methods[preset_name]()
    #     else:
    #         available_presets = ', '.join(preset_methods.keys())
    #         self.logger.error(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to 0-360 range.
        
        Args:
            angle: Input angle in degrees (can be any value)
            
        Returns:
            float: Normalized angle in range [0, 360)
            
        Examples:
            _normalize_angle(-10) -> 350
            _normalize_angle(370) -> 10
            _normalize_angle(-370) -> 350
        """
        # Use modulo operation to normalize angle
        normalized = angle % 360
        
        # Ensure the result is positive (modulo can return negative for some negative inputs)
        if normalized < 0:
            normalized += 360
            
        return normalized