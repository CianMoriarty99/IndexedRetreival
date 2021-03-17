import math

class Retrieve:
    
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index,termWeighting,pseudoRelevanceFeedback):
        self.index = index
        self.termWeighting = termWeighting
        self.pseudoRelevanceFeedback = pseudoRelevanceFeedback
        
        #Run the calculation for collection size once at the beginning and then store it.
        self.collectionSizeStore = self.collectionSize()
        
        

    #A method for calculating the document frequency of a given term
    def docFreq(self, term):
        if term in self.index:
            return len(self.index[term])
        else :
            return 1
    
    #A method for calculating the total number of documents in the collection 
    def collectionSize(self):
        n = 0
        for i in self.index:
            for v in self.index[i]:
                if v > n:
                    n = v
                    
        return n
    

    #A method for calculating the idf value of a given term
    def idfCalc(self, term):
        return math.log((self.collectionSizeStore/self.docFreq(term)),10)
        

        

    #A method performing retrieval for specified query
    def forQuery(self, query): 
        
        idx = self.index
        tw = self.termWeighting
        prf = self.pseudoRelevanceFeedback
        
        #Enable this print to visualise how the query changes with PRF
        #print(query)


            
        #A dictionary containing a document ID, and the weight of query words it contains
        candidateDocuments = dict()
        #A dictionary containing a document ID, and the length of the document
        docLengthDict = dict()
        
        #Search through the documents that contain a word from the query
        for q in query:

            try: 
                
                #for the document IDs contained within the queried index
                for docIDs in idx[q]:
                    
                    #In the case of binary 
                    #only increment the weight for each candidate document length by 1 
                    #without considering repeated query occurences within the same document
                    if(tw == 'binary'):
                        if docIDs in candidateDocuments.keys():
                            candidateDocuments[docIDs] += 1 
                        else :
                            candidateDocuments[docIDs] = 1 
                            
                    #In the case of tf 
                    #increment the weight accounting for the number of repeated query occurences within the same document
                    if(tw == 'tf'):
                        if docIDs in candidateDocuments.keys():
                            candidateDocuments[docIDs] += idx[q][docIDs] * query[q]
                        else :
                            candidateDocuments[docIDs] = idx[q][docIDs] * query[q]
                     
                            
                    #In the case of tf-idf
                    #Calculate the idf for each query word within a candidate document
                    #and multiply it with the tf weight to get the tf-idf weight
                    if(tw == 'tfidf'):
                        if docIDs in candidateDocuments.keys():
                            candidateDocuments[docIDs] += idx[q][docIDs] * query[q] * self.idfCalc(q)
                        else :
                            candidateDocuments[docIDs] = idx[q][docIDs] * query[q] * self.idfCalc(q)
                            
                            
                        

            except KeyError:
                continue
                
            
        #Search through all the word keys in the index
        #if the candidate document ID is in the dictionary of index keys
        #increment the document length dictionary according to weight scheme
        #in a similar fashion to the process above
        for keys in idx:
            for docIDs in idx[keys]:
                if docIDs in candidateDocuments.keys():
                
                    if(tw == 'binary'):
                        try:
                            docLengthDict[docIDs] += 1
    
                        except KeyError:
                            docLengthDict[docIDs] = 1
                            
                    if(tw == 'tf'):    
                        try:
                            docLengthDict[docIDs] += idx[keys][docIDs] ** 2
    
                        except KeyError:
                            docLengthDict[docIDs] = idx[keys][docIDs] ** 2
                        
                    if(tw == 'tfidf'):

                        try:
                            docLengthDict[docIDs] += (idx[keys][docIDs] * self.idfCalc(keys)) ** 2
    
                        except KeyError:
                            docLengthDict[docIDs] = (idx[keys][docIDs] * self.idfCalc(keys)) ** 2 
                    


        #A dictionary that stores a documentID and its corresponding similarity value
        simDict = dict()
        
        #For each documentID in the document length dictionary            
        for docIDs in docLengthDict:
            
                #The similarity is calculated from the given cosine similarity equation
                similarity =  candidateDocuments[docIDs] / math.sqrt(docLengthDict[docIDs])
                #Store the similarity for each individual document, for later ranking
                simDict[docIDs] = similarity
            
        
        
        #Sort the documents in the dictionary by their similarity value in descending order
        rankedDocs = sorted(simDict, key=simDict.get, reverse=True)
        
        
        
        if(prf == True):
            n = 3
            t = 3

            #Perform tfidf search and return top n documents
            topN = rankedDocs[:n]
            
            topT = dict()
            
            #look through each document
            for keys in idx:
                for docIDs in idx[keys]:
                    if docIDs in topN:
                        
                        try:
                            #look through the whole index for document ids
                            #create a dict of Words and tfidf values
                            topT[keys] += (idx[keys][docIDs] * self.idfCalc(keys)) ** 2 
                        
                        except KeyError:
                            topT[keys] = (idx[keys][docIDs] * self.idfCalc(keys)) ** 2 

            rankedKeys = sorted(topT, key=topT.get, reverse=False)
                                
            topT = rankedKeys[:t]
            for t in topT:
                query.update({t:1})

            
            #Need to toggle prf so that the inner query call doesn't recurse forever
            self.pseudoRelevanceFeedback = False
            rankedDocs = self.forQuery(query)
            
                
            
    

            
        #Ensure it is set back to default for the next fresh query   
        self.pseudoRelevanceFeedback = prf
        #Return a list of ranked documentIDs
        return rankedDocs 
            

        
            




