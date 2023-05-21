from settings.dynamic import DynamicSettings
from settings.static import StaticSettings


class AppSettings:
    static_settings: StaticSettings = StaticSettings()
    dynamic_settings: DynamicSettings = DynamicSettings
