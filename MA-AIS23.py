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
    
    lengthMessage = 15 # Amought of AIS messages of one MMSI
    timeIntervall = 30 # Maximum time intervall between two AIS messages in Sec
    rSize = 10000 # radius of the circ in m
    speedMin = 0 # Speed value in km/h
    speedMax = 55.0 # Speed value in km/h (55 km/h ~ 30 kn)
    minMove = 10 # minimal Movement in m
    precision = 1 # digets behind dot
    numberOfTrackVariations = 1 #17 #50 # how many variations of a track are includet in the result

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
        # del elements
        #gc.collect()
        # for element in elements:
        #    track.append(element)
        # dataset.append(track)

        
        
        dataset, preTime = wgs84toProjection(dataset, startTime, preTime) # add [5] x and [6] y by Mercator projection
        # statcount = fileStats(dataset, path, statcount, startTime, preTime)
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
        # statcount = fileStats(dataset, path, statcount, startTime, preTime)
        #for track in dataset:
        #    if 7500 <= len(track) <= 100000 and 30 <= (np.sqrt((float(track[len(track)-1][4]-track[0][4]))**2+(float(track[len(track)-1][5]-track[0][5])**2))):
        #        print(track[0][0])
        #        print(len(track))
        #        print(np.sqrt((float(track[len(track)-1][4]-track[0][4]))**2+(float(track[len(track)-1][5]-track[0][5])**2)))
        #        print('--')
        #dataset = roundTracks(dataset, precision) # rounds 'x' and 'y' values to 'presission' digets behind dot
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

    # elements = [] #[0]*len(aisDataList)
    # elementCounter = 0
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
            # element[2] = int(element[2])
            # element[3] = float(element[3])
            # element[4] = float(element[4])
            # tmpElement = [element[i] for i in [2, 0, 3, 4]]
    
            # tmpElement[0:3] = element[2:5] # MMSI, LAT, LOG
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
    # print('Zeilen gelöscht: \t' + str((len(aisDataList))-elementCounter))
    # free up memory
    del aisDataList

    # resizedElements = ['0'] * elementCounter
    # resizedElements = elements[:elementCounter]
    

    # print(elements)
    # print(resizedElements)

    
    # print('inputFile Zeilen: \t' + str(len(resizedElements)))
    preTime = statusMsg(startTime, preTime)

    # free up memory
    # del aisDataList
    del element
    del tmpElement
    # del elements
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
        # if posElement % 1000000 == 0: 
        #     print('pE:\t' + str(posElement) + '\tlE:\t' + str(len(elements)) + '\tpoped:\t' + str(poped))
        # check if lat is out of boundery AND check if long is out of boundery
        if latMin <= float(elements[posElement][2]) <= latMax and lonMin <= float(elements[posElement][3]) <= lonMax:
            # check is success - append to new list
            tmpElements.append(elements[posElement])
            # tmpElements[elementCounter] = elements[posElement]
            # elementCounter = elementCounter + 1
            
        posElement = posElement + 1 

    # resizedElements = ['0'] * elementCounter
    # resizedElements = tmpElements[:elementCounter]

    print('elementCutting Zeilen im Bereich: \t' + str(len(tmpElements)))
    preTime = statusMsg(startTime, preTime)

    # free up memory  
    del elements
    del posElement
    del latMax
    del latMin
    del lonMax
    del lonMin
    # del tmpElements
    
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
            #elements[posElement][1] = time.mktime(datetime.strptime(elements[posElement][1], '%d/%m/%Y %H:%M:%S').timetuple())
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
            #print(elementPso)
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

    #print('MaxSpeed: \t' + '{0:.3f}'.format(speedMax))

    while posTrack < len(dataset): # select track
        posElement = 1
        while posElement < len(dataset[posTrack]): # select element
            if (np.sqrt((float(dataset[posTrack][posElement][2])-float(dataset[posTrack][posElement-1][2]))**2 + (float(dataset[posTrack][posElement][3])-float(dataset[posTrack][posElement-1][3]))**2)/(int(dataset[posTrack][posElement][1])-int(dataset[posTrack][posElement-1][1])) <= speedMax):
            #
                curTrackLen = curTrackLen + 1
            # else:
            #     print(dataset[posTrack][posElement-1]) 
            #     print(dataset[posTrack][posElement])
                
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
        # if distance between first an last track element is greater than 'minMove' m
        if (np.sqrt((float(dataset[posTrack][len(dataset[posTrack])-1][2])-float(dataset[posTrack][0][2]))**2 + (float(dataset[posTrack][len(dataset[posTrack])-1][3])-float(dataset[posTrack][0][3]))**2)) >= minMove:

            tmpDataset.append(dataset[posTrack])
        #     print('Movment: ' + str(np.sqrt((float(track[len(track)-1][2])-float(track[0][2]))**2+(float(track[len(track)-1][3])-float(track[0][3]))**2)))
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
        # print(dataset[posTrack][0])
        XDelta = dataset[posTrack][0][2] 
        YDelta = dataset[posTrack][0][3]
        #print(posTrack, YDelta,posTrack)
        while posElement < len(dataset[posTrack]):
            dataset[posTrack][posElement][2] = dataset[posTrack][posElement][2] - XDelta # Calculation of element on x-axis relative to previouse element ('elementPre')
            dataset[posTrack][posElement][3] = dataset[posTrack][posElement][3] - YDelta # Calculation of element on y-axis relative to previouse element ('elementPre')
            # print('track:\t'  + '{0:.3f}'.format(dataset[posTrack][posElement][3]) + '\tYDelta:\t' + '{0:.3f}'.format(YDelta) + '\tresult:\t' + '{0:.3f}'.format(dataset[posTrack][posElement][3] - YDelta))

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
                # if (dataset[posTrack][lenTrack][2] + offsetX)**2 > sCirc**2:
                #     print('xL:\t' + '{0:.3f}'.format(dataset[posTrack][lenTrack][2]) + '\toffX:\t' + '{0:.3f}'.format(offsetX))
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
                # print(len(dataset[posTrack]))
                while posElement < len(dataset[posTrack]):
                    tmpTrack[posElement][2] = dataset[posTrack][posElement][2] + offsetX
                    tmpTrack[posElement][3] = dataset[posTrack][posElement][3] + offsetY
                    # if np.sqrt((tmpTrack[posElement][2])**2 + (tmpTrack[posElement][3])**2) > rSize:
                    #     print('{0:.3f}'.format(np.sqrt((tmpTrack[posElement][2])**2 + (tmpTrack[posElement][3])**2)))
                    #     print('offsetX:\t' + '{0:.3f}'.format(offsetX) + '\toffsetY:\t' + '{0:.3f}'.format(offsetY))
                    #     print('XTrack:\t' + '{0:.3f}'.format(tmpTrack[posElement][2] - offsetX) + '\tYTrack:\t' + '{0:.3f}'.format(tmpTrack[posElement][3] - offsetY))
                    #     print('ResXTrack:\t' + '{0:.3f}'.format(tmpTrack[posElement][2]) + '\tResYTrack:\t' + '{0:.3f}'.format(tmpTrack[posElement][3]))

                    posElement = posElement + 1
                    # print(trackVarcount)
                    # print(posTrack)
                    # print(posTrack*numberOfTrackVariations + trackVarcount)
                    # print(tmpTrack)
                posElement = 0
                # print(len(tmpTrack))
                # print(tmpTrack)
                tmpDataset.append(tmpTrack)
                # tmpTrack = []

            
            
            trackVarcount = trackVarcount + 1
    
        posTrack = posTrack + 1


    
    
    
    print('circRanPosTracks Repso:\t' + str(len(dataset)) + ' * ' + str(numberOfTrackVariations) + ' = ' +  str(len(tmpDataset)))
    preTime = statusMsg(startTime, preTime)

    # print(len(tmpDataset))
    # print(len(tmpDataset[-1]))
    # i = 0
    # for track in tmpDataset:
    #     print(str(i) + '\t' + str(len(track)))
    #     i = i + 1


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
    
    print(dataset[0][0])

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
                #print(str(dataset[posTrack][posElement][2]) + '\t\t' + str(dataset[posTrack][posElement][3]) + '\t\t' + str(math.sqrt((dataset[posTrack][posElement][2])**2 + (dataset[posTrack][posElement][3]**2))) + '\t\t'+ str(dataset[posTrack][posElement][1]))
                #print('center distance: ' + str(centerDistance))
                curTrackLen = curTrackLen + 1
            # else:
            #     print(dataset[posTrack][posElement])
            #     print(centerDistance)
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
    # lenDataset = len(dataset)

    # # print(dataset)
    # # print(lenDataset)

    # # tmpDataset = [0] * (8*lenDataset)
    # # tmpDataset1 = [0] * lenDataset
    # # tmpDataset2 = [0] * lenDataset
    # # tmpDataset3 = [0] * lenDataset
    # # tmpDataset4 = [0] * lenDataset
    # # tmpDataset5 = [0] * lenDataset
    # # tmpDataset6 = [0] * lenDataset
    # # tmpDataset7 = [0] * lenDataset
    # # tmpDataset8 = [0] * lenDataset
    # tmpDataset = []
    # # tmpDataset= []
    # # tmpDataset2= []
    # # tmpDatasetk3= []
    # # tmpDataset4= []
    # # tmpDataset5= []
    # # tmpDataset6= []
    # # tmpDataset7= []
    # # tmpDataset8= []
    # posTrack = 0
    # posElement = 0

    # tmpDataset = copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset) + copy.deepcopy(dataset)
    # # tmpDataset2 = copy.deepcopy(dataset)
    # # tmpDataset3 = copy.deepcopy(dataset)
    # # tmpDataset4 = copy.deepcopy(dataset)
    # # tmpDataset5 = copy.deepcopy(dataset)
    # # tmpDataset6 = copy.deepcopy(dataset)
    # # tmpDataset7 = copy.deepcopy(dataset)
    # # tmpDataset8 = copy.deepcopy(dataset)

    # # print(type(tmpTrack1))
    # # print(type(tmpDataset))

    # while posTrack < lenDataset:
    #     while posElement < len(dataset[posTrack]):
    #         # # print(tmpDataset[1*lenDataset + posTrack][posElement][4])
    #         tmpDataset[1*lenDataset + posTrack][posElement][4] = -1 * (dataset[posTrack][posElement][4])
    #         # # print(tmpDataset[1*lenDataset + posTrack][posElement][4])
    #         # # df
    #         tmpDataset[2*lenDataset + posTrack][posElement][4] = -1 * (dataset[posTrack][posElement][4])
    #         tmpDataset[2*lenDataset + posTrack][posElement][5] = -1 * (dataset[posTrack][posElement][5])
    #         tmpDataset[3*lenDataset + posTrack][posElement][5] = -1 * (dataset[posTrack][posElement][5])
            
    #         tmpDataset[4*lenDataset + posTrack][posElement][4] =  1 * (dataset[posTrack][posElement][5])
    #         tmpDataset[4*lenDataset + posTrack][posElement][5] =  1 * (dataset[posTrack][posElement][4])
    #         tmpDataset[5*lenDataset + posTrack][posElement][4] = -1 * (dataset[posTrack][posElement][5])
    #         tmpDataset[5*lenDataset + posTrack][posElement][5] =  1 * (dataset[posTrack][posElement][4])
    #         tmpDataset[6*lenDataset + posTrack][posElement][4] = -1 * (dataset[posTrack][posElement][5])
    #         tmpDataset[6*lenDataset + posTrack][posElement][5] = -1 * (dataset[posTrack][posElement][4])
    #         tmpDataset[7*lenDataset + posTrack][posElement][4] =  1 * (dataset[posTrack][posElement][5])
    #         tmpDataset[7*lenDataset + posTrack][posElement][5] = -1 * (dataset[posTrack][posElement][4])

    #         posElement = posElement + 1
    #     posElement = 0
    #     # tmpDataset.aconda install conda-forge::tensorflowppend(tmpDataset1)
    #     # tmpDataset.append(tmpDataset2)
    #     # tmpDataset.append(tmpDataset3)
    #     # tmpDataset.append(tmpDataset4)

    #     # tmpDataset.append(tmpDataset5)
    #     # tmpDataset.append(tmpDataset6)
    #     # tmpDataset.append(tmpDataset7)
    #     # tmpDataset.append(tmpDataset8)
    #     # tmpDataset.extend((tmpDataset1, tmpDataset2, tmpDataset3, tmpDataset4, tmpDatasetk5, tmpDataset6, tmpDataset7, tmpDataset8))
        
        
        
    #     posTrack = posTrack + 1

    # # print(len(tmpTrack1))
    # # print(tmpTrack1[0]) #[0:len(tmpTrack1)])
    # # sdf
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset1[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset2[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset3[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset4[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset1[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset1[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset1[posTrack])
    # #     posTrack = posTrack + 1
    # # posTrack = 0
    # # while posTrack < len(dataset):
    # #     tmpDataset.append(tmpDataset1[posTrack])
    # #     posTrack = posTrack + 1


    # # tmpDataset.extend((tmpTrack1[0:], tmpTrack2[0:], tmpTrack3[0:], tmpTrack4[0:], tmpTrack5[0:], tmpTrack6[0:], tmpTrack7[0:], tmpTrack8[0:]))


    # # tmpDataset[0:lenDataset] = tmpTrack1 #+ tmpTrack2+ tmpTrack3+ tmpTrack4+ tmpTrack5+ tmpTrack6+ tmpTrack7+ tmpTrack8)
    # # tmpDataset[lenDataset:2*lenDataset] = tmpTrack2
    # # tmpDataset[2*lenDataset:3*lenDataset] = tmpTrack3
    # # tmpDataset[3*lenDataset:4*lenDataset] = tmpTrack4
    # # tmpDataset[4*lenDataset:5*lenDataset] = tmpTrack5
    # # tmpDataset[5*lenDataset:6*lenDataset] = tmpTrack6
    # # tmpDataset[6*lenDataset:7*lenDataset] = tmpTrack7
    # # tmpDataset[7*lenDataset:8*lenDataset] = tmpTrack8

    # # print(type(tmpDataset))

    # print('trackMirrow:\t' + str(len(dataset)) + ' * 8 = ' + str(len(tmpDataset)))
    # preTime = statusMsg(startTime, preTime)
    
    # # free up memory
    # del dataset
    # # del tmpTrack1
    # # del tmpTrack2
    # # del tmpTrack3
    # # del tmpTrack4
    # # del tmpTrack5
    # # del tmpTrack6
    # # del tmpTrack7
    # # del tmpTrack8
    # del posTrack
    # del posElement
    
    # return tmpDataset, preTime

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
    # tmpTrack1= []
    # tmpTrack2= []
    # tmpTrack3= []
    # tmpTrack4= []
    # tmpTrack5= []
    # tmpTrack6= []
    # tmpTrack7= []
    # tmpTrack8= []
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
        # tmpTracks.append(tmpTrack1)
        # tmpTracks.append(tmpTrack2)
        # tmpTracks.append(tmpTrack3)
        # tmpTracks.append(tmpTrack4)

        # tmpTracks.append(tmpTrack5)
        # tmpTracks.append(tmpTrack6)
        # tmpTracks.append(tmpTrack7)
        # tmpTracks.append(tmpTrack8)
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


        # print(tracks)
        # print(len(tracks))
        # print(len(tracks[0]))
        # print(tracks[0][0])
        
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

# def fileStats(dataset, path, statcount, startTime, preTime):

#     file_path = path[:len(path)-4] + str(statcount) +'_tracks_state.csv'

#     textList = []
#     lengthList = []
#     timeList = []
    
#     textStr = ''
#     deltaLength = 0
#     length = 0
#     trackSpeed = 0
#     posTrack = 0

#     sumStepLength = 0
#     sumLength = 0
#     sumTime = 0
#     sumSpeed = 0
#     sumElements = 0
#     sumLenDataset = 0
#     posElement = 1


#     while posTrack < len(dataset): # select track
#         posElement = 1
#         while posElement < len(dataset[posTrack]): # select element 
#             deltaLength =np.sqrt((float(dataset[posTrack][posElement][4])-float(dataset[posTrack][posElement-1][4]))**2 + (float(dataset[posTrack][posElement][5])-float(dataset[posTrack][posElement-1][5]))**2)

#             lengthList.append(deltaLength)
#             timeList.append(float(dataset[posTrack][posElement][1])-float(dataset[posTrack][posElement-1][1]))
#             length = length + deltaLength
            
#             posElement = posElement + 1
#         timE = float(dataset[posTrack][posElement-1][1] - float(dataset[posTrack][0][1]))
#         trackSpeed = length / timE 
#         stepLength = length / (posElement)
#         sumStepLength = sumStepLength + stepLength
#         sumLength = sumLength + length
#         sumTime = sumTime + timE
#         sumSpeed = sumSpeed + trackSpeed
#         sumLenDataset = sumLenDataset + posElement
#         #sumElements = len(dataset[posTrack]) + sumElements 
#         textList.append(str('MMSI:\t ' + str(dataset[posTrack][0][0]) + '\t lenTrack:\t' + '{0:9.0f}'.format(len(dataset[posTrack])) +  '\tavgStepLength:\t' + '{0:0.6f}'.format(stepLength) + '\t lenght:\t' + '{0:9.3f}'.format(length) + '\t Time:\t' + '{0:9.0f}'.format(timE) + '\t trackAvgSpeed:\t' + '{0:3.6f}'.format(trackSpeed)))
#         #textList.append(str(str(dataset[posTrack][0][0]) + ', \t' + '{0:9.0f}'.format(len(dataset[posTrack])) +  ', \t' + '{0:0.6f}'.format(stepLength) + ', \t' + '{0:9.3f}'.format(length) + ', \t' + '{0:9.0f}'.format(timE) + ', \t' + '{0:3.6f}'.format(trackSpeed)))
        
#         deltaLength = 0
#         length = 0

#         posTrack = posTrack + 1




#     avgStepLength = sumStepLength / len(dataset)
#     avgLength = sumLength / len(dataset)
#     avgTime = sumTime / len(dataset)
#     avgSpeed = sumSpeed / len(dataset)
#     avgLenDataset = sumLenDataset / len(dataset)

    
    

#     textList.append(str('Dataset:\t' + '{0:9.0f}'.format(len(dataset)) + '\avgLenDataset:\t' + '{0:6.3f}'.format(avgLenDataset) + '\tavgStepLength:\t' + '{0:3.6f}'.format(avgStepLength) + '\tavgLength:\t '+ '{0:9.3f}'.format(avgLength) + '\tavgTime:\t' + '{0:9.3f}'.format(avgTime) + '\tavgSpeed:\t' + '{0:3.6f}'.format(avgSpeed)))
#     #textList.append('{0:9.0f}'.format(len(dataset)) + ', \t' + '{0:6.3f}'.format(avgLenDataset) + ', \t' + '{0:3.6f}'.format(avgStepLength) + ', \t' + '{0:9.3f}'.format(avgLength) + ', \t' + '{0:9.3f}'.format(avgTime) + ', \t' + '{0:3.6f}'.format(avgSpeed))

#     # Median
#     # auartil1 = df.time_diff.quantile([0.25])
#     # auartil2 = df.time_diff.quantile([0.5])
#     # auartil3 = df.time_diff.quantile([0.75])

#     lengthListMean = statistics.mean(lengthList)
#     lengthListStdev = statistics.stdev(lengthList)
#     lengthListQantiles = statistics.quantiles(lengthList, n=4)
#     lengthListMin = min(lengthList)
#     lengthListMax = max(lengthList)


#     textList.append(str('lengthMin:\t' + '{0:9.0f}'.format(lengthListMin) + '\tlength1.Quartil:\t' + '{0:6.3f}'.format(lengthListQantiles[0]) + 
#                          '\tlengthMedi:\t' + '{0:3.6f}'.format(lengthListQantiles[1]) + '\tlengthMean:\t '+ '{0:9.3f}'.format(lengthListMean) + 
#                          '\tlength3.Quartil:\t' + '{0:9.3f}'.format(lengthListQantiles[2]) + '\tlengthMax:\t' + '{0:3.6f}'.format(lengthListMax)) + '\tlengthstdev: \t' + '{0:3.6f}'.format(lengthListStdev))
    
#     # textList.append(str('{0:9.0f}'.format(lengthListMin) + '{0:6.3f}'.format(lengthListQantiles[0]) + 
#     #                      '{0:3.6f}'.format(lengthListQantiles[1]) +  '{0:9.3f}'.format(lengthListMean) + 
#     #                      '{0:9.3f}'.format(lengthListQantiles[2]) + '{0:3.6f}'.format(lengthListMax)) + '{0:3.6f}'.format(lengthListStdev))


    

#     timeListMean = statistics.mean(timeList)
#     timeListStdev = statistics.stdev(timeList)
#     timeListQantiles = statistics.quantiles(timeList, n=4)
#     timeListMin = min(timeList)
#     timeListMax = max(timeList)

#     textList.append(str('timeMin:\t' + '{0:9.0f}'.format(timeListMin) + '\ttime1.Quartil:\t' + '{0:6.3f}'.format(timeListQantiles[0]) + 
#                          '\ttimeMedi:\t' + '{0:3.6f}'.format(timeListQantiles[1]) + '\ttimeMean:\t '+ '{0:9.3f}'.format(timeListMean) + 
#                          '\ttime3.Quartil:\t' + '{0:9.3f}'.format(timeListQantiles[2]) + '\ttimeMax:\t' + '{0:3.6f}'.format(timeListMax)) + '\ttimestdev: \t' + '{0:3.6f}'.format(timeListStdev))
    
#     # textList.append(str('{0:9.0f}'.format(timeListMin) + '{0:6.3f}'.format(timeListQantiles[0]) + 
#     #                      '{0:3.6f}'.format(timeListQantiles[1]) +  '{0:9.3f}'.format(timeListMean) + 
#     #                      '{0:9.3f}'.format(timeListQantiles[2]) + '{0:3.6f}'.format(timeListMax)) + '{0:3.6f}'.format(timeListStdev))
    
    
    

#     textStr = '\n'.join(map(str, textList))
#     # print(textStr)

#     # creating a new empty file or overwriting existend file
#     save_file = open(file_path, 'w')
#     save_file.write(textStr)
#     save_file.close() 

#     statcount = statcount + 1 


#     print('fileStats')
#     preTime = statusMsg(startTime, preTime)

#     # free up memory
#     del textList
#     del textStr
#     del deltaLength
#     del length 
#     del avgSpeed
#     del posTrack
#     del posElement

#     return statcount

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
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-10.csv')
# mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-11.csv')
#mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-08-10.csv')


#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_Test.csv')

# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_filtered.csvconda install conda-forge::tensorflow')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000.csv')
mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000_s.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk-2023-11-11_266288000_xs.csv')
# mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_Test.csv')

#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\AIS_170155704388163013_2729-1701557044518_miss.csv')

#mainAIS('C:\\Dokumente\\Studium\\Jade-HS\\MA-ET\\AIS-MA\\aisdk_20060302_test_s.csv')
    
#mainAIS('/home/sebastian/Dokumente/MA-ET/AIS-MA/AIS_170155704388163013_2729-1701557044518_miss.csv')   

# ToDo
# - RAM Nutzung optimieren

# Test #2