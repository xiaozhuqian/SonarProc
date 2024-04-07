function messages = decodejsf2(filepath)
    %decode a jsf file. See https://www.edgetech.com/pdfs/ut/048-EdgeTech-JSF-File-Format.pdf for format details
    %filepath: char vector or scalar string specifying the full path of the file to decode
    %messages: a structure array with fields
    %    - protocol
    %    - sessionid
    %    - messagetype
    %    - commandtype
    %    - subsytem
    %    - channel
    %    - sequencenumber
    %    - content: the content of the message. For messagetype other than 80, it is the raw bytes of the message
    %For message type 80, it is a another structure with many fields, including
    %        - sonardata: the decoded data
    %
    %code (c) 2019 G. de Sercey
    %licensed under BSD
    %Redistribution and use in source and binary forms, with or without
    %modification, are permitted provided that the above copyright notice is retained.
    
    [fid, errmsg] = fopen(filepath, 'r', 'l');
    assert(fid > 0, 'failed to open file "%s" with error message: %s', filepath, errmsg);
    cleanup = onCleanup(@()fclose(fid));
    messages = [];
    fseek(fid, 0, 'eof');    %locate end of file and keep length around. Can't use feof as it's triggered only after you've attempted to go past the eof.
    eof = ftell(fid);
    fseek(fid, 0, 'bof');
    while ftell(fid) < eof 
        messages = [messages, decodenextmessage(fid)]; %#ok<AGROW> We don't know how many messages there will be
    end
end

function message = decodenextmessage(fid)
    %decode the next message in the jsf file
    persistent msgheader;
    if isempty(msgheader), msgheader = messageheaderdescription(); end
    message = decoderecord(fid, msgheader);
    assert(~isempty(message.marker) && message.marker == 5633, 'Invalid message marker encountered');
    switch message.messagetype  %at present only messagetype 80 is properly decoded
        case 80
            message.content = decodesonardata(fid, message.datasize);
        otherwise  %other message types just keep as raw bytes
            message.content = fread(fid, [1 message.datasize], '*uint8');
    end
end

function sonarmessage = decodesonardata(fid, datasize)
    %decode message type 80
    persistent sonarheader;
    if isempty(sonarheader), sonarheader = sonarheaderdescription(); end
    sonarmessage = decoderecord(fid, sonarheader);
    totalsamples = sonarmessage.samplecount * (sonarmessage.dataformat + 1);  %dataformat 0 = 1 short per sample, 1 = 2 short per sample
    assert(totalsamples * 2 == datasize - 240, 'Mismatch between number of samples and message size');
    rawdata = fread(fid, [1, totalsamples], 'int16');
    %note that possibly sonarmessage.header.msb should be used to expand the rawdata to 20 bits.
    %I've not understood how that works so ignoring it here
    sonarmessage.sonardata = rawdata * 2^(-sonarmessage.weightingfactor);
end

function record = decoderecord(fid, description)
    %decode a record according to the given description
    %description is a table with 3 variables: Name, Type, Count. Each row of the table is a record field (in order)
    %Name specifies the field name to use in the output record. A Name of '' implies that the field is ignored
    %Type is the type of the field to read, any precision accepted by fread is valid
    %Count is the length of the field in unit of Type
    data = rowfun(@(count, type) fread(fid, [1 count], type), description, 'InputVariables', {'Count', 'Type'}, 'ExtractCellContent', true, 'OutputFormat', 'cell');
    assert(~feof(fid), 'End of file encountered inside a record');
    tokeep = ~strcmp(description.Name, '');
    record = cell2struct(data(tokeep), description.Name(tokeep), 1);
end

function msgheader = messageheaderdescription()
    msgheader = cell2table({
            'marker'        1       'uint16'
            'protocol'      1       'uint8'
            'sessionid'     1       'uint8'
            'messagetype'   1       'uint16'
            'commandtype'   1       'uint8'
            'subsystem'     1       'uint8'
            'channel'       1       'uint8'
            'sequence'      1       'uint8'
            ''              1       'uint16'
            'datasize'      1       'uint32'
            }, 'VariableNames', {'Name', 'Count', 'Type'});
    checkdescription(msgheader, 16);
end

function sonarheader = sonarheaderdescription()
        sonarheader = cell2table({
            'pingtime'                  1   'int32'
            'startdepth'                1   'uint32'
            'pingnumber'                1   'uint32'
            ''                          2   'int16'
            'msb'                       1   '*uint16'   %keep as uint16 as bits are important. May be critical
            ''                          5   'int16'
            'idcode'                    1   'int16'
            'validityflag'              1   '*uint16'  %keep as uint16 as bits are important
            ''                          1   'uint16'
            'dataformat'                1   'int16'     %critical field
            'antennadistance_aft'       1   'int16'
            'antennadistance_starboard' 1   'int16'
            ''                          2   'int16'
            'kilometerpipe'             1   'single'
            ''                          16  'int16'
            'X'                         1   'int32'
            'Y'                         1   'int32'
            'coordunits'                1   'int16'
            'annotation'                24  '*char'
            'samplecount'               1   'uint16'    %critical field
            'sampleinterval'            1   'uint32'
            'gainfactor'                1   'uint16'
            'transmitlevel'             1   'int16'
            ''                          1   'int16'
            'startfrequency'            1   'uint16'
            'endfrequency'              1   'uint16'
            'sweeplength'               1   'uint16'
            'pressure'                  1   'int32'
            'depth'                     1   'int32'
            'samplefrequency'           1   'uint16'
            'identifier'                1   'uint16'
            'altitude'                  1   'int32'
            ''                          2   'int32'
            'cpuyear'                   1   'int16'
            'cpuday'                    1   'int16'
            'cpuhour'                   1   'int16'
            'cpuminute'                 1   'int16'
            'cpusecond'                 1   'int16'
            'cpubasis'                  1   'int16'
            'weightingfactor'           1   'int16'     %critical field
            'pulses'                    1   'int16'
            'heading'                   1   'uint16'
            'pitch'                     1   'int16'
            'roll'                      1   'int16'
            'temperature'               1   'int16'
            ''                          1   'int16'
            'triggersource'             1   'int16'
            'marknumber'                1   'uint16'
            'nmeahour'                  1   'int16'
            'nmeaminute'                1   'int16'
            'nmeasecond'                1   'int16'
            'nmeacourse'                1   'int16'
            'nmeaspeed'                 1   'int16'
            'nmeaday'                   1   'int16'
            'nmeayear'                  1   'int16'
            'milliseconds'              1   'uint32'
            'maxadc'                    1   'uint16'
            ''                          2   'int16'
            'softversion'               6   '*char'
            'corrfactor'                1   'int32'
            'packetnumber'              1   'uint16'
            'addecimation'              1   'int16'
            'fftdecimation'             1   'int16'
            'watertemperature'          1   'int16'
            'layback'                   1   'single'
            ''                          1   'int32'
            'cableout'                  1   'uint16'
            ''                          1   'uint16'
            }, 'VariableNames', {'Name', 'Count', 'Type'});
    checkdescription(sonarheader, 240);
end

function checkdescription(description, expectedsize)
    %helper function to check that mistakes haven't been made in record descriptions
    simplifiedtypes = regexp(description.Type, '[a-tv-z][a-z]+\d*', 'match', 'once'); %get rid of starting * and u and any possible => operation
    validtypes = {'int', 'int8', 'int16', 'int32', 'int64', 'char', 'short', 'long', 'single', 'double', 'float', 'float32'};
    matchingsize = [4, 1, 2, 4, 8, 1, 2, 4, 4, 8, 4, 4];
    [found, where] = ismember(simplifiedtypes, validtypes);
    fieldoffsets = zeros(height(description), 1);
    fieldoffsets(found) = matchingsize(where);
    fieldoffsets = cumsum([0; fieldoffsets .* description.Count]);
    if ~all(found) || fieldoffsets(end) ~= expectedsize
        description.FieldOffset = fieldoffsets(1:end-1);
        warning('Error in description %s', inputname(1));
        keyboard;  %enter debug mode
    end
end

