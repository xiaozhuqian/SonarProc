import numpy as np

def extract_from_mat(mat, frequency):
    messages = mat[list(mat.keys())[3]]

    #extract high or low frequency sonar messages
    acoustics = []
    for i in range(messages.shape[1]):  
        message = messages[0,i]
        if message['messagetype'] == 80:
            if message['subsystem'] == frequency:
                acoustics.append(message)
    
    #get the minimum ping number in extracted sonar messages
    ping_numbers = []  
    for i in range(len(acoustics)):
        high_ping_number = acoustics[i]['content']['pingnumber'][0,0][0,0]
        ping_numbers.append(high_ping_number)
    ping_numbers = list(set(ping_numbers))
    min_ping = min(ping_numbers)

    #extract sonardata, geo_coords, sonar attitudes, nmeaspeed/course.
    acous_del = acoustics.copy()
    ping_numbers_del = ping_numbers.copy()
    signals = [] #sonardata
    geo_coords = [] #lontitude, latitude; port*2, starboard*2
    attitudes = [] #heading, pitch, roll; port*3, starboard*3
    nmeas = [] #nmeaspeed, nmeacourse; port*2, starboard*2; not accurate because lacking LSB1 and LSB2
    error_ping_numbers = [] #pings lacking port or starboard message
    i = 1
    while len(acous_del):
        # if i >=3:
        #     break
        # print('acous-del lenght:', len(acous_del))
        # print('min_ping',min_ping)
        # print('i:',i)

        #find the minimum ping sonar data
        j = 1
        for value in acous_del: 
            ping_number = value['content']['pingnumber'][0,0][0,0]
            if ping_number == min_ping:
                if value['channel']: #starboard
                    signal_right = value['content']['sonardata'][0,0]
                    x_right = value['content']['X'][0,0][0,0]/10000./60.
                    y_right = value['content']['Y'][0,0][0,0]/10000./60.
                    heading_right = value['content']['heading'][0,0][0,0]/100.
                    pitch_right = value['content']['pitch'][0,0][0,0]/32768.0*180.0
                    roll_right = value['content']['roll'][0,0][0,0]/32768.0*180.0
                    nmeaspeed_right = value['content']['nmeaspeed'][0,0][0,0]/10.
                    nmeacourse_right = value['content']['nmeacourse'][0,0][0,0]

                    value_right = value
                else: #port
                    signal_left = value['content']['sonardata'][0,0][0]
                    signal_left = np.flipud(signal_left)[np.newaxis,:]
                    x_left = value['content']['X'][0,0][0,0]/10000./60.
                    y_left = value['content']['Y'][0,0][0,0]/10000./60.
                    heading_left = value['content']['heading'][0,0][0,0]/100.
                    pitch_left = value['content']['pitch'][0,0][0,0]/32768.0*180.0
                    roll_left = value['content']['roll'][0,0][0,0]/32768.0*180.0
                    nmeaspeed_left = value['content']['nmeaspeed'][0,0][0,0]/10.
                    nmeacourse_left = value['content']['nmeacourse'][0,0][0,0]

                    value_left = value
            j = j+1
        
        # check if port and stratport data exit
        if 'signal_left' in vars():
            flag_left = True
        else:
            flag_left = False
            error_ping_numbers.append(ping_number)
            # print(f'{ping_number} lose port message!')
            # logging.info(f'{ping_number} lose port message!')
        if 'signal_right' in vars():
            flag_right = True
        else:
            flag_right= False
            error_ping_numbers.append(ping_number)
            # print(f'{ping_number} lose starboard message!')
            # logging.info(f'{ping_number} lose starboard message!')

        # concate the minimum port and startport
        if flag_left and flag_right:  
            signal = np.concatenate((signal_left, signal_right), axis=1)
            signals.append(signal)
            geo_coord = [x_left, y_left, x_right, y_right]
            geo_coords.append(geo_coord)
            attitude = [heading_left, pitch_left, roll_left, heading_right, pitch_right, roll_right]
            attitudes.append(attitude)
            nmea = [nmeaspeed_left, nmeacourse_left, nmeaspeed_right, nmeacourse_right]
            nmeas.append(nmea)

        # update orginal dataset by removing the minimum ping data
        if flag_left:
            acous_del.remove(value_left)
            del value_left, signal_left, x_left, y_left, heading_left, pitch_left, roll_left, nmeaspeed_left, nmeacourse_left
        if flag_right:
            acous_del.remove(value_right)
            del value_right, signal_right, x_right, y_right, heading_right, pitch_right, roll_right, nmeaspeed_right, nmeacourse_right
        
        # update minimum ping number
        ping_numbers_del.remove(min_ping)
        if ping_numbers_del:
            min_ping = min(ping_numbers_del)
        #print(f'{i}th ping/{len(acoustics)/2}')
        #info(f'{idx}th image/{len(input_names)},{i}th ping/{len(acoustics)/2}, {input_name}')

        i=i+1
    
    signals = np.reshape(np.array(signals), (np.array(signals).shape[0], -1))
    remain_acoustics = len(acous_del)
    return signals, np.around(geo_coords,6), np.around(attitudes, 4), nmeas, \
    remain_acoustics, error_ping_numbers
    