class a(object): 
    
    def printb(self): 
        print(self.c()) 

    def c(self):
        retur 3                    

class b(a): 
    def printb(self): 
        super().printb() 

    def c(self):
        return 2                  

e = b()
e.printb()
