import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
import dash_daq as daq

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}
rootLayout = html.Div([
    daq.BooleanSwitch(
        on=True,
        id='darktheme-daq-booleanswitch',
        className='dark-theme-control'
    ), html.Br(),
    daq.ToggleSwitch(
        id='darktheme-daq-toggleswitch',
        className='dark-theme-control'
    ), html.Br(),
    daq.ColorPicker(
        value=dict(hex='#171717'),
        id='darktheme-daq-colorpicker',
        className='dark-theme-control'
    ), html.Br(),
    daq.Gauge(
        min=0,
        max=10,
        value=6,
        color=theme['primary'],
        id='darktheme-daq-gauge',
        className='dark-theme-control'
    ), html.Br(),
    daq.GraduatedBar(
        value=4,
        color=theme['primary'],
        id='darktheme-daq-graduatedbar',
        className='dark-theme-control'
    ), html.Br(),
    daq.Indicator(
        value=True,
        color=theme['primary'],
        id='darktheme-daq-indicator',
        className='dark-theme-control'
    ), html.Br(),
    daq.Knob(
        min=0,
        max=10,
        value=6,
        id='darktheme-daq-knob',
        className='dark-theme-control'
    ), html.Br(),
    daq.LEDDisplay(
        value="3.14159",
        color=theme['primary'],
        id='darktheme-daq-leddisplay',
        className='dark-theme-control'
    ), html.Br(),
    daq.NumericInput(
        min=0,
        max=10,
        value=4,
        id='darktheme-daq-numericinput',
        className='dark-theme-control'
    ), html.Br(),
    daq.PowerButton(
        on=True,
        color=theme['primary'],
        id='darktheme-daq-powerbutton',
        className='dark-theme-control'
    ), html.Br(),
    daq.PrecisionInput(
        precision=4,
        value=299792458,
        id='darktheme-daq-precisioninput',
        className='dark-theme-control'
    ), html.Br(),
    daq.StopButton(
        id='darktheme-daq-stopbutton',
        className='dark-theme-control'
    ), html.Br(),
    daq.Slider(
        min=0,
        max=100,
        value=30,
        targets={"25": {"label": "TARGET"}},
        color=theme['primary'],
        id='darktheme-daq-slider',
        className='dark-theme-control'
    ), html.Br(),
    daq.Tank(
        min=0,
        max=10,
        value=5,
        id='darktheme-daq-tank',
        className='dark-theme-control'
    ), html.Br(),
    daq.Thermometer(
        min=95,
        max=105,
        value=98.6,
        id='darktheme-daq-thermometer',
        className='dark-theme-control'
    ), html.Br()

])