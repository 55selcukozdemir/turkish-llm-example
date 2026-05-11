

class DataPreparer():
    def __init__(self, data_path):
        satirlar = []

        with open(data_path, "r", encoding="utf-8") as file:
            for satir in file:
                satirlar.append(satir.strip())

        self.satirlar = satirlar
    
    def get_list_of_array(self):
        return self.satirlar
    
    def get_split_batchs(self, batch_size):
        return [
            self.satirlar[i:i + batch_size]
            for i in range(0, len(self.satirlar), batch_size)
        ]
