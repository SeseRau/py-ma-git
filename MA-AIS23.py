import csv # CSV Things
import os  # OS absolute filepath
import time # localtime to unix time
from datetime import datetime  # localtime to unix time
from datetime import date # localtime to unix time
import math # wgs84 to xy
import numpy as np # wgs84 to projection
import random # randomise dataset by withe noise and reelementing
import psutil # RAM values
import copy
import statistics # statistics

# therms
# AIS-Messages = element single parameter of AIS-Message  [UUID,time,lat,long]
# AIS-Messages = position whole AIS-Message [UUID,time,lat,long]
# AIS-Messages = positions [UUID#1,time,lat,long], [UUID#2,time,lat,long], [UUID#3,time,lat,long]
# AIS-Messages sorted to one individuel vessel = track [UUID#1,time,lat,long], [UUID#1,time,lat,long], [UUID#1,time,lat,long]
# AIS-Messages sorted to many individuel vessel = dataset [UUID#1,time,lat,long], [UUID#1,time,lat,long], [UUID#3,time,lat,long], [UUID#3,time,lat,long],


def mainAIS(path):
    '''Steuert das aufrufen der einzelnen '''
    print('Startzeit:\t', time.ctime(time.time()) + ' ' + path)
    startTime = time.time() # timestemp at start
    preTime = time.time() # timestemp at the previous function
     # initializing the titles and elements list
    dataset = []
    track = []
    
    lengthMessage = 100 # Amought of AIS messages of one MMSI
    timeIntervall = 30 # Maximum time intervall between two AIS messages in Sec
    rSize = 10000 # radius of the circ in m
    speedMin = 0 # Speed value in km/h
    speedMax = 55.0 # Speed value in km/h (55 km/h ~ 30 kn)
    minMove = 10 # minimal Movement in m
    precision = 1 # digets behind dot
    numberOfTrackVariations = 1 #17 #50 # how many variations of a track are included in the result

    #aisDK
    latMin = 53.0 # minimal latitude
    latMax = 65.0# maximal latitude
    lonMin = 0.0 # minimal longditude
    lonMax = 22.0 # maximal longditude

    statcount = 0 # counts state reports
    
    primeSortingElement = 0 # prime sorting element of elements [IMO]
    secSortingElement = 1 # prime sorting element of elements [time]
    
  
    elements, preTime = inputFile(path, startTime, preTime) # read's file and convert it into nested list with each element (AIS-Message) (IMOnr, CET, Lat, Lon)
    
    if len(elements) != 0:
        # check if 'elements' is empty 
        elements, preTime = elementCutting(elements, latMin, latMax, lonMin, lonMax, startTime, preTime) # delete's misselemented dataset AISdk
        elements, preTime = timeConversion(elements, startTime, preTime) # convert timestamp from CET (Central European Time) to UnixTime (seconds since 01.01.1970)
        elements, preTime = elementsSorting(elements, primeSortingElement, secSortingElement, startTime, preTime) # sorting AIS-Messgages by 'MMSI' assending and by time assending
        elements, preTime = dualElement(elements, startTime, preTime) # deletes 'element' with the same UUID and timestamp
        dataset, preTime = trackBuild(elements, lengthMessage, timeIntervall, startTime, preTime) # checks if the AIS-Messages belong to the same IMOnr and are less than 'timeIntervall' secounds apart and if there are at least 'minMessages' AIS-Messages from one IMOnr

        
        
        dataset, preTime = wgs84toProjection(dataset, startTime, preTime) # add [5] x and [6] y by Mercator projection

        dataset, preTime = trackSplitting(dataset, lengthMessage, startTime, preTime) # checks if Message number per Track is surfissiant and splites too long 'tracks' into parts
        
        
        dataset, preTime = speedControl(dataset, speedMin, speedMax, timeIntervall, lengthMessage, startTime, preTime) # checks if elementchange in time is valide and for 'timeIntervall'
        
        dataset, preTime = movementControl(dataset, minMove, startTime, preTime)
        
        dataset, preTime = centerPosTracks(dataset, startTime, preTime) # reset start element to 0,0
        dataset, preTime = circRanPosTracks(dataset, rSize, numberOfTrackVariations, startTime, preTime) # reset start element in relation to direction on edge of a circ
        dataset, preTime = roundTracks(dataset, precision, startTime, preTime) # rounds x and y values to avoid fake pressison
        dataset, preTime = circCutTracks(dataset, rSize, lengthMessage, startTime, preTime) # cuts a circ into the 'dataset'
        dataset, preTime = lenghtCheck(dataset, lengthMessage, startTime, preTime) # checks if dataset have the lenght 'lengthMessage'
        dataset, preTime = trackMirrow(dataset, startTime, preTime) # mirrows track by x and y axis and does the same after switching x and y value 1 track -> 8 track
        dataset, preTime = polarCoord(dataset, precision, startTime, preTime) # adds polar coordinates

        fileSaving(dataset, path, lengthMessage, numberOfTrackVariations, startTime, preTime) # writes conclusion into .csv
        
    # 'elements' is empty
    else:
        # error message
        print('Datei ist fehlerhaft!')
    print('Skript Ende --- %s seconds ---' % (time.time() - startTime)) # print runtime
    
def inputFile(path, startTime, preTime):
    element = []
    elements = []
    # tmpElement = [] #'['0']*4
    preElement = [] #'['0']

    

    #read file 
    aisDataFile = open(path, 'r')
    aisDataList = aisDataFile.readlines()
    aisDataFile.close()


    posElement = 0
    if(aisDataList[0].split(',')[2] == 'MMSI'):
        # First line in file is discription -> skip first line
        posElement = 1
        
    # data refinement from line reed file to nested list with [MMSI, Zeit, Lat, Log]
    while posElement < len(aisDataList): 
        # tmpElement = ['0']*5
        element = aisDataList[posElement].split(',')
        
        if element != preElement: # filter duplications
        
            tmpElement = ['0']*4

            tmpElement[0] = int(element[2]) #MMSI
            tmpElement[1] = str(element[0]) #TIME
            tmpElement[2] = float(element[3]) #LAT
            tmpElement[3] = float(element[4]) #LOG
            
            if (str(element[1]) == 'Class A' and float(tmpElement[2]) != 91.0 and float(tmpElement[3]) != 181.0): # check for filter parameters           
                elements.append(tmpElement)
                # elementCounter = elementCounter + 1
            
        preElement = element
        posElement = posElement + 1

    preTime = statusMsg(startTime, preTime)
    print('Zeilen gelesen: \t' + str(len(aisDataList)))
    # free up memory
    del aisDataList

    preTime = statusMsg(startTime, preTime)

    # free up memory
    del element
    del tmpElement

    return elements, preTime
 
def elementCutting(elements, latMin, latMax, lonMin, lonMax, startTime, preTime):
    '''Loescht AIS-Nachrichten, welche ausserhalb des ausgewaehlten Quadrates liegen.'''
    tmpElements = [] #[0] * len(elements)
    
    if latMax <= latMin:
        print('latMax <= latMin')
    if lonMax <= lonMin:
        print('lonMax <= lonMin')

    # elementCounter = 0
    posElement = 0
    while posElement < len(elements):
  
        # check if lat is out of boundery AND check if long is out of boundery
        if latMin <= float(elements[posElement][2]) <= latMax and lonMin <= float(elements[posElement][3]) <= lonMax:
            # check is success - append to new list
            tmpElements.append(elements[posElement])
            
            
        posElement = posElement + 1 



    print('elementCutting Zeilen im Bereich: \t' + str(len(tmpElements)))
    preTime = statusMsg(startTime, preTime)

    # free up memory  
    del elements
    del posElement
    del latMax
    del latMin
    del lonMax
    del lonMin

    
    return tmpElements, preTime

def timeConversion(elements, startTime, preTime):
    '''Wandelt das Datumsformat (DD/MM/YYYY HH:MM:SS) in Unix Zeit um'''
    '''Wandelt das Datumsformat (YYYY-MM-DDTHH:MM:SS) in Unix Zeit um'''

    # print(elements)
    
    tmpElement = []
    posElement = 0
   
    if elements[1][1][2] == '/':
         # timeformat (DD/MM/YYYY HH:MM:SS)

        while posElement < len(elements):
            tmpElement = elements[posElement][1]     
            tmpElement = tmpElement.replace(':','/')
            tmpElement = tmpElement.replace(' ','/')
            tmpElement = list(map(int,tmpElement.split('/')))
            tmpElement = datetime(tmpElement[2], tmpElement[1], tmpElement[0], tmpElement[3], tmpElement[4], tmpElement[5])
            elements[posElement][1] = int(tmpElement.timestamp() )
            
            posElement = posElement + 1

    elif elements[1][1][4] == '-':
        # timeformat is (YYYY-MM-DDTHH:MM:SS)

        posElement = 0
        while posElement < len(elements):
            elements[posElement][1] = time.mktime(datetime.strptime(elements[posElement][1], '%Y-%m-%d%H:%M:%S').timetuple())
            
            posElement = posElement + 1

    print('timeConversion Zeit gewandelt: \t' + str(len(elements)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    del tmpElement
    del posElement 
    
    return elements, preTime

def elementsSorting(elements, primeSortingElement, secSortingElement, startTime, preTime):
    '''Ordnet die einzelnen AIS-Nachrichten nach MMSI (aufsteigend) und nach UnixTime (aussteigend)'''
    tmpElement = []
    
    # sort by MMSI (assending) and UnixTime (assending)
    tmpElement = sorted(elements, key=lambda x: (x[primeSortingElement], x[secSortingElement]))

    print('elementsSorting Elemente sortiert: \t' + str(len(tmpElement)))
    preTime = statusMsg(startTime, preTime)

    # free up memory        
    del elements
    
    return tmpElement, preTime

def dualElement(elements, startTime, preTime):
    '''Sortiert AIS Messages mit gleicher MMSI und Zeitstempel aus - duplikate'''
    tmpElements = []
    posElement = 0
    
    # check if MMSI and timestemp of elements are equal 
    while posElement < len(elements):
        
        if elements[posElement][0] != elements[posElement-1][0] or elements[posElement][1] != elements[posElement-1][1]:
            # MMSI and timestemp of two messages are equal
            tmpElements.append(elements[posElement])
            
        posElement = posElement + 1

    print('dualElement AIS-Nachricht geprueft: \t' + str(len(tmpElements)))
    preTime = statusMsg(startTime, preTime)

    # free up memory    
    del elements
    del posElement
    
    return tmpElements, preTime
    
def trackBuild(elements, lengthMessage, timeIntervall, startTime, preTime):

    startElement = 0
    track = []
    dataset = []

    elementPos = 1
    sameMMSI = 1
    
    while elementPos < len(elements):

        if elements[elementPos-1][0] == elements[elementPos][0] and (elements[elementPos][1] - elements[elementPos-1][1]) <= timeIntervall:
            # add to track
            sameMMSI = sameMMSI + 1
        elif sameMMSI >= lengthMessage:
            # build track
            for k in range(startElement, startElement + sameMMSI):
                    track.append(elements[k]) 
            startElement = startElement + sameMMSI + 1
            sameMMSI = 0
            dataset.append(track)
            track = [] 
        elif sameMMSI < lengthMessage:
            # track to short
            startElement = startElement + sameMMSI + 1
            sameMMSI = 0
        else:
            # no track
            startElement = startElement + 1
            sameMMSI = 0
        
        if  sameMMSI >= lengthMessage and elementPos == (len(elements)-1):
            # if last element of elements belongs to track
             # build track
            for k in range(startElement, startElement + sameMMSI):
                    track.append(elements[k]) 
            startElement = startElement + sameMMSI + 1
            sameMMSI = 0
            dataset.append(track)
            track = [] 


        elementPos = elementPos + 1

    

    print('TrackBuild Tracks gebildet: \t' + str(len(dataset)))
    preTime = statusMsg(startTime, preTime)


    # free up memory        
    del track
    del elements
    del startElement
    del sameMMSI
    del elementPos
    
    return dataset, preTime


def wgs84toProjection(dataset, startTime, preTime):
    '''Wandelt WGS84 Koordinaten (polar) in eine Projektion und fuegt die neuen Koordinaten an das Ende jeder AIS-Nachricht an'''
    
    allLon = []
    allLat = []

    track = []

    tmpXY = []
    
    
    # build list of every element
    
    posTrack = 0
    posElement = 0

    while posTrack < len(dataset):
        while posElement < len(dataset[posTrack]):
            allLon.append(dataset[posTrack][posElement][3])
            allLat.append(dataset[posTrack][posElement][2])
            posElement = posElement + 1
        posElement = 0
        posTrack = posTrack + 1

    maxLamda = max(allLon)
    minLamda = min(allLon)
    
    mitteLamda = (float(maxLamda) - float(minLamda))/2 + float(minLamda)
    
    


    posTrack = 0
    posElement = 0

    constant = 6378137*np.pi / 180

    # tmpDataset = copy.deepcopy(dataset)
    

    while posTrack < len(dataset):
        while posElement < len(dataset[posTrack]):

            
            #long ->x
            tmpX = ((constant*np.cos(math.radians(float(dataset[posTrack][posElement][2]))))*(float(dataset[posTrack][posElement][3]) -mitteLamda))

            #lat ->y
            dataset[posTrack][posElement][2] = (111320*(float(dataset[posTrack][posElement][2])))
            
            dataset[posTrack][posElement][3] = tmpX
            

            posElement = posElement + 1
        
        posElement = 0
        posTrack = posTrack+ 1
    
    
            
    print('wgs84toProjection Tracks umgerechnet: \t' + str(len(dataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    del allLon

    
    
    return dataset, preTime


    

def trackSplitting(dataset, lengthMessage, startTime, preTime):
    '''Prueft ob eine Track eine bestimmte Länge erreicht und teile ihn, falls er ueber diese Länge hinausgeht. Verwirft unvollständige Endstücke.'''
    tmpDataset = []
    posTrack = 0

    while posTrack < len(dataset):
        if(len(dataset[posTrack])) > lengthMessage:
            totalTrackParts = int((len(dataset[posTrack])) / lengthMessage)
            curTrackParts = 0
            while curTrackParts < totalTrackParts:
                tmpDataset.append(dataset[posTrack][(lengthMessage*curTrackParts):(lengthMessage*(curTrackParts+1))])
                curTrackParts = curTrackParts + 1
        elif(len(dataset[posTrack])) == lengthMessage:
                 tmpDataset.append(dataset[posTrack])
        posTrack = posTrack + 1
    
    
    print('trackSplitting Gebildete Tracks: \t' + str(len(dataset)) +' x ' + str(lengthMessage) + ' = ' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory        
    del dataset
    del posTrack
    del totalTrackParts
    del curTrackParts
    
    return tmpDataset, preTime

 

def speedControl(dataset, speedMin, speedMax, timeIntervall, lengthMessage, startTime, preTime):
    '''Um AIS-Nachrichten mit fehlerhafter Zeit und Positionen zu entfernen'''
    
    
    tmpDataset = []
    curTrackLen = 1

    posTrack = 0
    posElement = 1

    speedMax = speedMax / 3.6 # conversion km/h to m/s


    while posTrack < len(dataset): # select track
        posElement = 1
        while posElement < len(dataset[posTrack]): # select element
            if (np.sqrt((float(dataset[posTrack][posElement][2])-float(dataset[posTrack][posElement-1][2]))**2 + (float(dataset[posTrack][posElement][3])-float(dataset[posTrack][posElement-1][3]))**2)/(int(dataset[posTrack][posElement][1])-int(dataset[posTrack][posElement-1][1])) <= speedMax):
                curTrackLen = curTrackLen + 1

                
            posElement = posElement + 1
            
        if curTrackLen == lengthMessage:
            tmpDataset.append(dataset[posTrack])   
        curTrackLen = 1
        posElement = 0
        posTrack = posTrack + 1

    print('speedControl Geschw. gepruefte: \t' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory 
    del dataset
    
    return tmpDataset, preTime
        
def  movementControl(dataset, minMove, startTime, preTime):
    
    tmpDataset = []
    
    # for every 'track'
    posTrack = 0

    while posTrack < len(dataset):
        if (np.sqrt((float(dataset[posTrack][len(dataset[posTrack])-1][2])-float(dataset[posTrack][0][2]))**2 + (float(dataset[posTrack][len(dataset[posTrack])-1][3])-float(dataset[posTrack][0][3]))**2)) >= minMove:

            tmpDataset.append(dataset[posTrack])
        
        posTrack = posTrack + 1

    print('movementControl Strecken geprueft: \t' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory        
    del dataset
    
    return tmpDataset, preTime

def centerPosTracks(dataset, startTime, preTime):
    '''Rechnet Koordinatensystem um (km) und zentriert das erste Element auf (x.y) (0.0)'''
    #dataset = [[[MMSI, UnixTime, Lat, Lon, x, y], [Element 2], [Element3]], [[ELement4], [Element5], [Element6]]]

    
    
    XDelta = -2
    YDelta = -2

    posTrack = 0
    posElement = 0

    while posTrack < len(dataset):

        XDelta = dataset[posTrack][0][2] 
        YDelta = dataset[posTrack][0][3]

        while posElement < len(dataset[posTrack]):
            dataset[posTrack][posElement][2] = dataset[posTrack][posElement][2] - XDelta # Calculation of element on x-axis relative to previouse element ('elementPre')
            dataset[posTrack][posElement][3] = dataset[posTrack][posElement][3] - YDelta # Calculation of element on y-axis relative to previouse element ('elementPre')
 
            posElement = posElement + 1
        posElement = 0
        posTrack = posTrack + 1

          
            
    print('centerPosTracks Resize: \t\t' + str(len(dataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory  
    
    return dataset, preTime

def circRanPosTracks(dataset, rSize, numberOfTrackVariations, startTime, preTime):
    '''Verschiebt die eines Tracks zufaellig auf einem Kreis'''

    # print(len(dataset[0]))
    # print(dataset[0])
    
    ranDis = 0 # random dessision
    offsetX = 0
    offsetY = 0
    sCirc = rSize - 0.0 #small circ

    posTrack = 0
    posElement = 0

    tmpTrack = []
    tmpDataset = []
    
    

    while posTrack < len(dataset):
        trackVarcount = 0
        
        while trackVarcount < numberOfTrackVariations:

            
            
            lenTrack = len(dataset[posTrack]) - 1
            if np.sqrt((dataset[posTrack][lenTrack][2])**2 + (dataset[posTrack][lenTrack][3])**2) < sCirc and abs(dataset[posTrack][lenTrack][2]) < sCirc and abs(dataset[posTrack][lenTrack][3]) < rSize:
                offsetX = 0
                offsetY = 0
                
                
                tmpVar = abs(np.sqrt((sCirc)**2 - (dataset[posTrack][lenTrack][3])**2) - abs(dataset[posTrack][lenTrack][2]))
                offsetX = random.uniform(- tmpVar, tmpVar)

                tmpVar = abs(np.sqrt((sCirc)**2 - (abs(dataset[posTrack][lenTrack][2]) + abs(offsetX))**2)  - abs(dataset[posTrack][lenTrack][3]))
                offsetY =  random.uniform(- tmpVar, tmpVar)

                if np.sqrt((dataset[posTrack][lenTrack][2] + offsetX)**2 + (dataset[posTrack][lenTrack][3] + offsetY)**2) > rSize or np.sqrt((offsetX)**2 + (offsetY)**2) > rSize:
                    print('{0:.3f}'.format(np.sqrt((dataset[posTrack][lenTrack][2] + offsetX)**2 + (dataset[posTrack][lenTrack][3] + offsetY)**2)))
                    print('offsetX:\t' + '{0:.3f}'.format(offsetX) + '\toffsetY:\t' + '{0:.3f}'.format(offsetY))
                    print('lastXTrack:\t' + '{0:.3f}'.format(dataset[posTrack][lenTrack][2]) + '\tlastYTrack:\t' + '{0:.3f}'.format(dataset[posTrack][lenTrack][3]))
    
                if np.sqrt((dataset[posTrack][lenTrack][2] + offsetX)**2 + (dataset[posTrack][lenTrack][3] + offsetY)**2) > rSize or np.sqrt((offsetX)**2 + (offsetY)**2) > rSize :
                    print('offsetX:\t' + '{0:.3f}'.format(offsetX) + '\toffsetY:\t' + '{0:.3f}'.format(offsetY))
                    print('lastXTrack:\t' + '{0:.3f}'.format(dataset[posTrack][lenTrack][2]) + '\tlastYTrack:\t' + '{0:.3f}'.format(dataset[posTrack][lenTrack][3]))
                tmpTrack = copy.deepcopy(dataset[posTrack])

                while posElement < len(dataset[posTrack]):
                    tmpTrack[posElement][2] = dataset[posTrack][posElement][2] + offsetX
                    tmpTrack[posElement][3] = dataset[posTrack][posElement][3] + offsetY

                    posElement = posElement + 1

                posElement = 0

                tmpDataset.append(tmpTrack)


            
            
            trackVarcount = trackVarcount + 1
    
        posTrack = posTrack + 1


    
    
    
    print('circRanPosTracks Repso:\t' + str(len(dataset)) + ' * ' + str(numberOfTrackVariations) + ' = ' +  str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)



    # free up memory
    del offsetX
    del offsetY
    del posTrack
    del posElement
    del sCirc
    del ranDis
    del tmpTrack
    del dataset
    
    return tmpDataset, preTime

def roundTracks(dataset, precision, startTime, preTime):
    posTrack = 0
    posElement = 0
    
    while posTrack < len(dataset):
        while posElement < len(dataset[posTrack]):
            dataset[posTrack][posElement][2] = round(dataset[posTrack][posElement][2], precision)
            dataset[posTrack][posElement][3] = round(dataset[posTrack][posElement][3], precision)

            posElement = posElement + 1
        posTrack = posTrack + 1
        posElement = 0
        
    print('roundTracks: \t' + str(len(dataset)))
    preTime = statusMsg(startTime, preTime)

    return dataset, preTime

def circCutTracks(dataset, rSize, lengthMessage, startTime, preTime):
    '''Schneidet die Trackdaten auf den vom Schleppsonar aus sichtbaren Kreis zu'''
    tmpDataset = []
    curTrackLen = 0
    
    posTrack = 0
    posElement = 0

    while posTrack < len(dataset):
        while posElement < len(dataset[posTrack]):
            centerDistance = math.sqrt((dataset[posTrack][posElement][2])**2 + (dataset[posTrack][posElement][3]**2))
            if(centerDistance <= (rSize)): 
                curTrackLen = curTrackLen + 1
            posElement = posElement + 1
            
        if curTrackLen == lengthMessage:
            tmpDataset.append(dataset[posTrack])
        curTrackLen = 0
        posElement = 0
        posTrack = posTrack + 1
            
    print('circCutTracks Angezeigte AIS-Tracks: \t' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    del dataset
    
    return tmpDataset, preTime


def lenghtCheck(dataset, lengthMessage, startTime, preTime):

    tmpDataset = []
    
    failedLengthCheck = 0
    posTrack = 0


    while posTrack < len(dataset):
        # if track has the lenght 'lengthMessage'
        if len(dataset[posTrack]) == lengthMessage:
            tmpDataset.append(dataset[posTrack])
        else:
            failedLengthCheck = failedLengthCheck + 1
        #     print(len(dataset[posTrack]))
            
        posTrack = posTrack + 1

    if failedLengthCheck > 0:
        print('failedLengthCheck: \t' + str(failedLengthCheck))
      

    print('lenghtCheck: \t' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)
    
    # free up memory  
    del dataset

    return tmpDataset, preTime

def trackMirrow(dataset, startTime, preTime):
    # '''Erzeugt weitere Tracks durch Spiegelung an der X- und Y-Achse'''
 
    lenDataset = len(dataset)

    tmpDataset = []
    tmpTrack1 = [0] * lenDataset
    tmpTrack2 = [0] * lenDataset
    tmpTrack3 = [0] * lenDataset
    tmpTrack4 = [0] * lenDataset
    tmpTrack5 = [0] * lenDataset
    tmpTrack6 = [0] * lenDataset
    tmpTrack7 = [0] * lenDataset
    tmpTrack8 = [0] * lenDataset
    posTrack = 0
    posElement = 0


    while posTrack < len(dataset):

        tmpTrack1 = copy.deepcopy(dataset[posTrack])
        tmpTrack2 = copy.deepcopy(dataset[posTrack])
        tmpTrack3 = copy.deepcopy(dataset[posTrack])
        tmpTrack4 = copy.deepcopy(dataset[posTrack])
        tmpTrack5 = copy.deepcopy(dataset[posTrack])
        tmpTrack6 = copy.deepcopy(dataset[posTrack])
        tmpTrack7 = copy.deepcopy(dataset[posTrack])
        tmpTrack8 = copy.deepcopy(dataset[posTrack])
        while posElement < len(dataset[posTrack]):
            tmpTrack2[posElement][2] = -1 * (dataset[posTrack][posElement][2])
            tmpTrack3[posElement][2] = -1 * (dataset[posTrack][posElement][2])
            tmpTrack3[posElement][3] = -1 * (dataset[posTrack][posElement][3])
            tmpTrack4[posElement][3] = -1 * (dataset[posTrack][posElement][3])

            tmpTrack5[posElement][2] =  1 * (dataset[posTrack][posElement][3])
            tmpTrack5[posElement][3] =  1 * (dataset[posTrack][posElement][2])
            tmpTrack6[posElement][2] = -1 * (dataset[posTrack][posElement][3])
            tmpTrack6[posElement][3] =  1 * (dataset[posTrack][posElement][2])
            tmpTrack7[posElement][2] = -1 * (dataset[posTrack][posElement][3])
            tmpTrack7[posElement][3] = -1 * (dataset[posTrack][posElement][2])
            tmpTrack8[posElement][2] =  1 * (dataset[posTrack][posElement][3])
            tmpTrack8[posElement][3] = -1 * (dataset[posTrack][posElement][2])

            posElement = posElement + 1
        posElement = 0
        tmpDataset.extend((tmpTrack1, tmpTrack2, tmpTrack3, tmpTrack4, tmpTrack5, tmpTrack6, tmpTrack7, tmpTrack8))
        
        posTrack = posTrack + 1

    print('trackMirrow:\t' + str(len(dataset)) + ' * 8 = ' + str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    del dataset
    del tmpTrack1
    del tmpTrack2
    del tmpTrack3
    del tmpTrack4
    del tmpTrack5
    del tmpTrack6
    del tmpTrack7
    del tmpTrack8
    del posTrack
    del posElement
    
    return tmpDataset, preTime

def polarCoord(dataset, precision, startTime, preTime):
    '''Ergänzt die Daten um zwei Stellen fuer das polar Koordinatensystem'''
    

    posTrack = 0
    posPosition = 0
    deg = 0

    

    while posTrack < len(dataset):
        while posPosition < len(dataset[posTrack]):
            deg = math.degrees(np.arctan2(dataset[posTrack][posPosition][2], dataset[posTrack][posPosition][3]))
            if  deg < 0:
                deg = deg + 360
            elif deg > 360:
                deg = deg - 360
            dataset[posTrack][posPosition].append(round(deg, precision))
            dataset[posTrack][posPosition].append(round(math.sqrt(dataset[posTrack][posPosition][2]**2 + dataset[posTrack][posPosition][3]**2), precision))
            # dataset[posTrack][posPosition][5] = round(dataset[posTrack][posPosition][7], precision)
            posPosition = posPosition + 1
        posPosition = 0
        posTrack = posTrack + 1
        

            
    print('polarCoord:\t' + str(len(dataset)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    del posTrack
    del posPosition
    
    return dataset, preTime

    
def fileSaving(dataset, path, lengthMessage, numberOfTrackVariations, startTime, preTime):
    '''Speichert die AIS-Nachrichten (MMSI, TIME, LAT, LOG, X, Y) eines Tracks in einer Datei unter dem aufgerufenen Dateinamen + _kkn.csv'''

    # print(len(dataset))
    # print(dataset)
    
    # if there is a 'dataset'
    if len(dataset) != 0:
        print('File saving Track')


        # creating a new path
        csv_path = path[:len(path)-4] + '_' + str(numberOfTrackVariations) + '_kkn.csv'
        gnu_path = path[:len(path)-4] + '_' + str(numberOfTrackVariations) + '_kkn_gnu.csv'

        #/home/sebastian/Dokumente/Python-Git/py-my

        # creates string for knn-file 
        lengthData = len(dataset)
        print('Laenge Datensatz: \t' + str(lengthData))

        blockSize = 2000

        csvElement = [0]*lengthMessage
        if lengthData < blockSize:
            csvTrack = [0]*lengthData
            blockSize = lengthData
        else:
            csvTrack = [0]*blockSize
        csvDataset = '0'
        
        posTrack = 0
        posCsvTrack = 0
        posElement = 0
    
        while posTrack < lengthData:
            while posElement < lengthMessage:
 
                #print('posTrack:\t' + str(posTrack) + '\tposElement:\t' + str(posElement))
                csvElement[posElement] = ','.join(map(str, dataset[posTrack][posElement]))
                posElement = posElement + 1
            csvTrack[posCsvTrack] = ','.join(map(str, csvElement))
            #gnuTracks = f'{gnuTracks}{gnuTrack}\n\n'
            posElement = 0
            posTrack = posTrack + 1
            posCsvTrack = posCsvTrack + 1
            
            if posTrack % blockSize == 0:
                # print('Aufbereitung CSV:\t' + str(posTrack) + '/' + str(lengthData))
                if posTrack != lengthData:
                    csvTrack.append(' ')
                csvDataset = '$'.join(map(str, csvTrack))
                if posTrack == blockSize:
                    print('CSV-Clear')
                    # creating a new empty file or overwriting existend file
                    save_file = open(csv_path, 'w')
                else:
                    # append file
                    save_file = open(csv_path, 'a')
                save_file.write(str(csvDataset))
                save_file.close()
                
                if (lengthData-posTrack) < blockSize:
                    csvTrack = [0]*(lengthData-posTrack)
                else:
                    csvTrack = [0]*blockSize
                posCsvTrack = 0
                    
        csvDataset = '$'.join(map(str, csvTrack))

        # creating a new empty file or overwriting existend file
        save_file = open(csv_path, 'a')
        save_file.write(str(csvDataset))
        save_file.close()
        
        print('KNN CSV --- %s seconds ---'  % (time.time() - startTime))
        preTime = statusMsg(startTime, preTime)
        print('.csv fuer KNN gespeichert')

        # free up memory
        del csvElement
        del csvTrack
        del csvDataset
        
        # creats sting for gnuplot-file
        # GNU Plot neeeds an other data formation
        gnuElement = [0]*lengthMessage
        if lengthData < blockSize:
           gnuTrack = [0]*lengthData
           blockSize = lengthData
        else:
           gnuTrack = [0]*blockSize
        gnuDataset = "0"
        tmpTracksList = []


        


        posTrack = 0
        posGnuTrack = 0
        posElement = 0
    
        while posTrack < lengthData:
           while posElement < lengthMessage:
 
               #print('posTrack:\t' + str(posTrack) + '\tposElement:\t' + str(posElement))
               gnuElement[posElement] = ', '.join(map(str, dataset[posTrack][posElement]))
               posElement = posElement + 1
           gnuTrack[posGnuTrack] = '\n'.join(map(str, gnuElement))
           #gnuTracks = f'{gnuTracks}{gnuTrack}\n\n'
           posElement = 0
           posTrack = posTrack + 1
           posGnuTrack = posGnuTrack + 1
           
           if posTrack % blockSize == 0:
               # print('Aufbereitung GNU:\t' + str(posTrack) + '/' + str(lengthData))
               
               if posTrack != lengthData:
                   gnuTrack.append(' ')
               gnuTracks = '\n\n'.join(map(str, gnuTrack))
               
               if posTrack == blockSize:
                   print('GNU-Clear')  
                   # creating a new empty file or overwriting existend file
                   save_file = open(gnu_path, 'w')
               else:
                   # append file
                   save_file = open(gnu_path, 'a')
               save_file.write(gnuTracks)
               save_file.close()
               if (lengthData-posTrack) < blockSize:
                   gnuTrack = [0]*(lengthData-posTrack)
               else:
                   gnuTrack = [0]*blockSize
               posGnuTrack = 0
                   
        gnuTracks = '\n\n'.join(map(str, gnuTrack))

        ## creating a new empty file or overwriting existend file
        save_file = open(gnu_path, 'a')
        save_file.write(gnuTracks)
        save_file.close()

        
        

        print('.csv fuer GNU gespeichert')
        preTime = statusMsg(startTime, preTime)

        # free up memory
        del gnuElement
        del gnuTrack
        del gnuDataset
        del dataset

        print('done')


def statusMsg(startTime, preTime):
    print('Laufzeit insg: ' + '{0:9.6f}'.format((time.time() - startTime)) + '\tLaufzeit Fkt: ' + '{0:6.6f}'.format((time.time() - preTime)) + '\tRAM in MiB: ' + '{0:6.1f}'.format(psutil.Process().memory_info()[1] / float(2**20)))
    
    preTime = time.time()
    return(preTime)

    
#----------------------Parameter-----------------------------------------
#mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdkRAWtest.csv')
#mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2024-04-07-xs.csv')
#mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-09.csv')
#mainAIS('/home/sebastian/Dokumente/AIS-FilesaisdkRAWtest_5_kkn/aisdk-2023-11-10.csv')
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-08.csv')
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-09.csv')
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-08-xs.csv')
mainAIS('/home/sebastian/Dokumente/Python-Git/py-ma-git/workdir/AIS-Files/aisdk-2023-11-08.csv')
#mainAIS('/home/sebastian/Dokumente/Python-Git/py-ma-git/workdir/AIS-Files/aisdk-2023-11-08-m.csv')
#mainAIS('/home/sebastian/Dokumente/Python-Git/py-ma-git/workdir/AIS-Files/aisdk-2023-11-08-s.csv')
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-11.csv')
#mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-08-10.csv')


#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_Test.csv')

# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_filtered.csvconda install conda-forge::tensorflow')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000_s.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000_xs.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_Test.csv')

#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_170155704388163013_2729-1701557044518_miss.csv')

#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk_20060302_test_s.csv')
    
#mainAIS('/home/sebastian/Dokumente/MA-ET/AIS-MA/AIS_170155704388163013_2729-1701557044518_miss.csv')   

# ToDo
# - RAM Nutzung optimieren

# Test #2