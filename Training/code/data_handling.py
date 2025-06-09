

class Txt_file:
    def __init__(self,store_path):
        self.store_path = store_path

        self.title = 'PARAMETERS'

    def save_parameters(self, par):
        file_path = self.store_path + f"/parameters.txt"

        with open(file_path, 'w') as file:
            file.write(f"{self.title}\n\n\n")
            for key, value in par.items():
                file.write(f"{key}: {value}\n")