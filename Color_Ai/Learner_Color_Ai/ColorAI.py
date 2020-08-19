import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

class Learn():
    def __init__(self):
        self.trained_data = pd.read_csv("learned_color_data.csv")
    

    
    def getColor(self, color_inp, data_ref):
        self.data = data_ref


        R = self.data["R"]
        G = self.data["G"]
        B = self.data["B"]

        X = self.data.iloc[:, :-2].values
        y = self.data["Id"]


        model = KNeighborsClassifier(n_neighbors = 27)
        model.fit(X, y)


        u_input = color_inp


        prediction = model.predict([u_input])

        self.prediction_index = np.where(self.data == prediction[0])[0][0]
        
        print("prediction:", self.data["Color name"][self.prediction_index])
        print("prediction_index:", self.data["Id"][self.prediction_index])
  
    
    
    def showDataFrame(self):
        print(self.trained_data)
    
    
    def quiz(self):
        n_test = 5
        n_correct = 0
        
        while n_test != 0:
            n_test -= 1
            
            question = input("test :").split(",")
            
            R = int(question[0])
            G = int(question[1])
            B = int(question[2])
            
            RGB = [R, G, B]
            
            self.getColor(RGB, self.trained_data)

            answer_status = input("answer status :")

            if answer_status == "C":
                n_correct += 1
            
        
            score = n_correct
            
            print("-" * 10 + "page" + "-" * 10)
        
        print("score :", score)
        
        
        new_score = {"Scores" : score}
        
        score_data = pd.read_csv("Scores.csv")
        new_score_data = score_data.append(new_score, ignore_index=True)
        
        new_score_data.to_csv("Scores.csv")
    
    
    
    def exam(self):
        given = pd.read_csv(input("exam sheet:"))
        
        questions = given["given"]
        
        for ask in questions:
            l_quest = ask.split(",")
            uinp = []
            for num in l_quest:
                uinp.append(int(num))
                
            self.getColor(uinp, self.trained_data)
        
        print(given["answer"])
        
        score = int(input("Score:"))
        n_test = int(input("n_test:"))
        data_to_append = {"n_test":n_test, "score":score}
        
        exam_score_data = pd.read_csv("exam_scores.csv")
        new_score_data = exam_score_data.append(data_to_append, ignore_index=True)
        
        new_score_data.to_csv("exam_scores.csv", index = False)
            
    
    def showExamScore(self):
        scores = pd.read_csv("exam_scores.csv")
        print(scores)
        
        plt.plot(scores["score"], "o", color = "orange")
        plt.plot(scores["score"], color = "green")
        plt.show()
        
    
    
    def showQuizScore(self):
        scores = score_data = pd.read_csv("Scores.csv")
        print(scores)
        
        plt.plot(scores["Scores"])
        plt.show()
            
    
    def teach(self):
        teach_status = "T"
        n_test = 0
        n_correct = 0
        n_wrong = 0
        
        while teach_status == "T":
            print("-" * 10 + str(n_test) +"-" * 10)
            
            n_test += 1
            
            uinp = input("color:").split(",")

            R = int(uinp[0])
            G = int(uinp[1])
            B = int(uinp[2])

            RGB = [R, G, B]

            self.getColor(RGB, self.trained_data)

            answer_status = input("answer status C/W:")

            if answer_status == "C":
                n_correct += 1
                
                shade_fam = self.data["Color name"][self.prediction_index]
                data_id = self.data["Id"][self.prediction_index]
                new_data = pd.DataFrame({"R":R, "G":G, "B":B, "Color name":shade_fam, "Id":data_id}, index = [0])
                
                self.trained_data = pd.concat([new_data, self.trained_data]).reset_index(drop = True)
                self.trained_data.to_csv("learned_color_data.csv", index=False)
                
                print(new_data)

            if answer_status == "W":
                n_wrong += 1
                
                add_learnings = input("Add New Lesson? Y/N :")
                
                if add_learnings == "Y":
                    shade_fam = input("shader family:")
                    data_id = int(input("new data id:"))
                    new_data = pd.DataFrame({"R":R, "G":G, "B":B, "Color name":shade_fam, "Id":data_id}, index = [0])
                    
                    self.trained_data = pd.concat([new_data, self.trained_data]).reset_index(drop = True)
                    self.trained_data.to_csv("learned_color_data.csv", index=False)
                             
            
            teach_status = input("teaching status T/F:")
            
            if teach_status == "F":
                print("-" * 10 + "teaching ended" + "-" * 10)
                print("number of tests : ", n_test)
                print("correct answer : ", n_correct)
                print("wrong answer : ", n_wrong)
                break


guide = pd.read_csv("learned_color_data.csv")
pd.set_option("display.max_rows", 500)
print(guide.shape)
