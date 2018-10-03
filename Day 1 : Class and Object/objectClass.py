# Class are being used to keep related things together


# Attributes

class Programmer:
    language = 'python'
    stream = 'Computer Science'
    jobPosition = 'Software Developer'

    def modify_language(self,new_language):
        self.language = new_language


#coder = Programmer()
#print(coder.language)

# Methods


coder = Programmer()

print (coder)

coder.modify_language("Java")
coder.modify_language("R")

print(coder.language)



class Programmer:

    def __init__(self,name):
        self.name = name

    def modify_language(self,new_language):
        self.language = new_language

coder1 = Programmer("Ruby")

print(coder1.name)

coder2 = Programmer("SAAS")

print(coder2.name)