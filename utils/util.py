import numpy as np

'''from https://github.com/OSOceanAcoustics/echopype/blob/main/echopype/utils/uwa.py'''
def calc_absorption(
    frequency,
    temperature=27,
    salinity=35,
    pressure=10,
    pH=8.1,
    sound_speed=None,
    formula_source="AM",
):
    """
    Calculate sea water absorption in units [dB/m].

    Parameters
    ----------
    frequency: int or numpy array
        frequency [Hz]
    temperature: num
        temperature [deg C]
    salinity: num
        salinity [PSU, part per thousand]
    pressure: num
        pressure [dbars]
    pH: num
        pH of water
    formula_source: str, {"AM", "FG", "AZFP"}
        Source of formula used to calculate sound speed.
        "AM" (default) uses the formula from Ainslie and McColm (1998).
        "FG" uses the formula from Francois and Garrison (1982).
        "AZFP" uses the the formula supplied in the AZFP Matlab code.
        See Notes below for the references.

    Returns
    -------
    Sea water absorption [dB/m].

    Notes
    -----
    Ainslie MA, McColm JG. (1998). A simplified formula for viscous
    and chemical absorption in sea water.
    The Journal of the Acoustical Society of America, 103(3), 1671–1672.
    https://doi.org/10.1121/1.421258

    Francois RE, Garrison GR. (1982). Sound absorption based on
    ocean measurements. Part II: Boric acid contribution and equation
    for total absorption.
    The Journal of the Acoustical Society of America, 72(6), 1879–1890.
    https://doi.org/10.1121/1.388673

    The accuracy of the simplified formula from Ainslie & McColm 1998
    compared with the original complicated formula from Francois & Garrison 1982
    was demonstrated between 100 Hz and 1 MHz.
    """
    if formula_source == "FG":
        f = frequency / 1000.0  # convert from Hz to kHz due to formula
        if sound_speed is None:
            c = 1412.0 + 3.21 * temperature + 1.19 * salinity + 0.0167 * pressure
        else:
            c = sound_speed
        A1 = 8.86 / c * 10 ** (0.78 * pH - 5)
        P1 = 1.0
        f1 = 2.8 * np.sqrt(salinity / 35) * 10 ** (4 - 1245 / (temperature + 273))
        A2 = 21.44 * salinity / c * (1 + 0.025 * temperature)
        P2 = 1.0 - 1.37e-4 * pressure + 6.2e-9 * pressure**2
        f2 = 8.17 * 10 ** (8 - 1990 / (temperature + 273)) / (1 + 0.0018 * (salinity - 35))
        P3 = 1.0 - 3.83e-5 * pressure + 4.9e-10 * pressure**2
        if np.all(temperature < 20):
            A3 = (
                4.937e-4
                - 2.59e-5 * temperature
                + 9.11e-7 * temperature**2
                - 1.5e-8 * temperature**3
            )
        else:
            A3 = (
                3.964e-4
                - 1.146e-5 * temperature
                + 1.45e-7 * temperature**2
                - 6.5e-10 * temperature**3
            )
        a = (
            A1 * P1 * f1 * f**2 / (f**2 + f1**2)
            + A2 * P2 * f2 * f**2 / (f**2 + f2**2)
            + A3 * P3 * f**2
        )
        sea_abs = a / 1000  # formula output is in unit [dB/km]

    elif formula_source == "AM":
        freq = frequency / 1000
        D = pressure / 1000
        f1 = 0.78 * np.sqrt(salinity / 35) * np.exp(temperature / 26)
        f2 = 42 * np.exp(temperature / 17)
        a1 = 0.106 * (f1 * (freq**2)) / ((f1**2) + (freq**2)) * np.exp((pH - 8) / 0.56)
        a2 = (
            0.52
            * (1 + temperature / 43)
            * (salinity / 35)
            * (f2 * (freq**2))
            / ((f2**2) + (freq**2))
            * np.exp(-D / 6)
        )
        a3 = 0.00049 * freq**2 * np.exp(-(temperature / 27 + D))
        sea_abs = (a1 + a2 + a3) / 1000  # convert to db/m from db/km

    elif formula_source == "AZFP":
        temp_k = temperature + 273.0
        f1 = 1320.0 * temp_k * np.exp(-1700 / temp_k)
        f2 = 1.55e7 * temp_k * np.exp(-3052 / temp_k)

        # Coefficients for absorption calculations
        k = 1 + pressure / 10.0
        a = 8.95e-8 * (1 + temperature * (2.29e-2 - 5.08e-4 * temperature))
        b = (
            (salinity / 35.0)
            * 4.88e-7
            * (1 + 0.0134 * temperature)
            * (1 - 0.00103 * k + 3.7e-7 * k**2)
        )
        c = (
            4.86e-13
            * (1 + temperature * (-0.042 + temperature * (8.53e-4 - temperature * 6.23e-6)))
            * (1 + k * (-3.84e-4 + k * 7.57e-8))
        )
        if salinity == 0:
            sea_abs = c * frequency**2
        else:
            sea_abs = (
                (a * f1 * frequency**2) / (f1**2 + frequency**2)
                + (b * f2 * frequency**2) / (f2**2 + frequency**2)
                + c * frequency**2
            )
    else:
        ValueError("Unknown formula source")

    return sea_abs

def absoption(frequency=600000, temperature=26, salinity=28, depth=5, pH=8.0):
    # frequency unit: Hz, temperature: deg C, salinity: [PSU, part per thousand], depth [m], absorb: db/m
    absorb = calc_absorption(frequency, 
                             temperature=temperature, 
                             salinity=salinity, 
                             pressure=depth, 
                             pH=pH, 
                             sound_speed=None, 
                             formula_source='AM') #pressure(dbar)=depth(m)
    return absorb