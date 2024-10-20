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
    
    lengthMessage = 3 # Amought of AIS messages of one MMSI
    timeIntervall = 300000 # 30 Maximum time intervall between two AIS messages in Sec
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
        # elements, preTime = elementCutting(elements, latMin, latMax, lonMin, lonMax, startTime, preTime) # delete's misselemented dataset AISdk
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

        preTime = stat(elements, dataset, startTime, preTime)
        # statcount = fileStats(dataset, path, statcount, startTime, preTime)
        # dataset, preTime = trackSplitting(dataset, lengthMessage, startTime, preTime) # checks if Message number per Track is surfissiant and splites too long 'tracks' into parts
        
        
        # dataset, preTime = speedControl(dataset, speedMin, speedMax, timeIntervall, lengthMessage, startTime, preTime) # checks if elementchange in time is valide and for 'timeIntervall'
        
        # dataset, preTime = movementControl(dataset, minMove, startTime, preTime)
        
        # dataset, preTime = centerPosTracks(dataset, startTime, preTime) # reset start element to 0,0
        # dataset, preTime = circRanPosTracks(dataset, rSize, numberOfTrackVariations, startTime, preTime) # reset start element in relation to direction on edge of a circ
        # dataset, preTime = roundTracks(dataset, precision, startTime, preTime) # rounds x and y values to avoid fake pressison
        # dataset, preTime = circCutTracks(dataset, rSize, lengthMessage, startTime, preTime) # cuts a circ into the 'dataset'
        # dataset, preTime = lenghtCheck(dataset, lengthMessage, startTime, preTime) # checks if dataset have the lenght 'lengthMessage'
        # dataset, preTime = trackMirrow(dataset, startTime, preTime) # mirrows track by x and y axis and does the same after switching x and y value 1 track -> 8 track
        # dataset, preTime = polarCoord(dataset, precision, startTime, preTime) # adds polar coordinates
        # statcount = fileStats(dataset, path, statcount, startTime, preTime)
        #for track in dataset:
        #    if 7500 <= len(track) <= 100000 and 30 <= (np.sqrt((float(track[len(track)-1][4]-track[0][4]))**2+(float(track[len(track)-1][5]-track[0][5])**2))):
        #        print(track[0][0])
        #        print(len(track))
        #        print(np.sqrt((float(track[len(track)-1][4]-track[0][4]))**2+(float(track[len(track)-1][5]-track[0][5])**2)))
        #        print('--')
        #dataset = roundTracks(dataset, precision) # rounds 'x' and 'y' values to 'presission' digets behind dot
        # fileSaving(dataset, path, lengthMessage, numberOfTrackVariations, startTime, preTime) # writes conclusion into .csv
        
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
        
        if element != 0: #preElement: # filter duplications
        
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


def stat(elements, dataset, startTime, preTime):

    # Anzahl Schiffe
    print('Anzahl Schiffe: \t' + str(len(dataset)))
    # print(dataset[0])       # Schiffe
    print(dataset[0][0])    # Schiffesposition
    # print(dataset[0][0][0]) # Schiffesposition MMSI

    # Zurückgelegte Strecke pro Schiff

    lengthPerShip = [0]*len(dataset)
    posTrack = 0
    while posTrack < len(dataset):
        posPosition = 1
        while posPosition < len(dataset[posTrack]):
            lengthPerShip[posTrack] = lengthPerShip[posTrack] + np.sqrt((dataset[posTrack][posPosition][2] - dataset[posTrack][posPosition-1][2])**2 + (dataset[posTrack][posPosition][3] - dataset[posTrack][posPosition-1][3])**2)
            posPosition = posPosition + 1
        posTrack = posTrack + 1

    # print(lengthPerShip)

    # Pos pro Schiff

    positionsPerShip = [0]*len(dataset)

    posTrack = 0
    while posTrack < len(dataset):
        positionsPerShip[posTrack]=len(dataset[posTrack])
        posTrack = posTrack + 1

    np.histogram(positionsPerShip, bins=range(0, 100, 1))


    # Dauer zwischen den Positionen

    i = 0
    totalNumPosDataset = 0
    while i < len(positionsPerShip):
        totalNumPosDataset = totalNumPosDataset + positionsPerShip[i]
        i = i + 1

    #print(totalNumPosDataset)
    #print(len(elements)-len(dataset))

    timeBetweenPos = [0]*(totalNumPosDataset-len(dataset))

    posTrack = 0
    totalPosCounter = 0
    while posTrack < len(dataset):
        posPosition = 1
        while posPosition < len(dataset[posTrack]):
            timeBetweenPos[totalPosCounter] = dataset[posTrack][posPosition][1] - dataset[posTrack][posPosition-1][1]
            posPosition = posPosition + 1
            totalPosCounter = totalPosCounter + 1
        posTrack = posTrack + 1

    #print("Zeit zwischen den Positionen: \t" + str(timeBetweenPos))
    print(statistics.mean(timeBetweenPos))


    # Richtung der Bewegung in Grad

    directionInDataset = [0]*(totalNumPosDataset-len(dataset))
    posTrack = 0
    totalPosCounter = 0
    while posTrack < len(dataset):
        posPosition = 1
        while posPosition < len(dataset[posTrack]):
            directionInDataset[totalPosCounter] = math.degrees(np.arctan2((dataset[posTrack][posPosition][2]-dataset[posTrack][posPosition-1][2]), (dataset[posTrack][posPosition-1][3]-dataset[posTrack][posPosition][3])))
            posPosition = posPosition + 1
            totalPosCounter = totalPosCounter + 1
        posTrack = posTrack + 1

    # print(directionInDataset)





    # Geschwindigkeit pro Schiff
    # Offene Frage - Wie berechnet man die Geschwindigkeit? - Nur Start und Endposition oder alle Positionen?


    speedInDataset = [0]*(totalNumPosDataset-len(dataset))
    posTrack = 0
    totalPosCounter = 0
    while posTrack < len(dataset):
        posPosition = 1
        while posPosition < len(dataset[posTrack]):
            speedInDataset[totalPosCounter] = (np.sqrt((dataset[posTrack][posPosition][2] - dataset[posTrack][posPosition-1][2])**2 + (dataset[posTrack][posPosition][3] - dataset[posTrack][posPosition-1][3])**2))/(dataset[posTrack][posPosition][1] - dataset[posTrack][posPosition-1][1])
            posPosition = posPosition + 1
            totalPosCounter = totalPosCounter + 1
        posTrack = posTrack + 1

    print(speedInDataset)
    return preTime




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
mainAIS('/home/sebastian/Dokumente/AIS-Files/aisdk-2023-11-08-6xs.csv')
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