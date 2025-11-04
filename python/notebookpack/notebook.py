class Note(object):
    def __init__(self, contents):
        self.contents = contents

    def get_number_of_lines(self):
        return self.contents.count("\n")

    def get_number_of_characters(self):
        return len(self.contents)

    def remove(self):
        self.contents = "삭제된 노트입니다."

    def __str__(self):
        return self.contents


class NoteBook(object):
    def __init__(self, name):
        self.name = name
        self.pages = 0
        self.notes = {}
        self.removed_pages = []

    def add_note(self, note, page_number=0):
        if len(self.notes.keys()) < 300:
            if page_number in self.removed_pages:
                print("삭제된 페이지입니다. 다른 페이지를 입력하세요")
                return
            if page_number == 0:
                while True:
                    if self.pages not in self.notes.keys():
                        self.notes[self.pages] = note
                        self.pages += 1
                        break
                    self.pages += 1
            else:
                if page_number not in self.notes.keys():
                    self.notes[page_number] = note
                else:
                    print("해당 페이지에는 이미 노트가 존재합니다.")
        else:
            print("더 이상 노트를 추가하지 못합니다.")

    def remove_note(self, page_number):
        del self.notes[page_number]
        self.removed_pages.append(page_number)
        # self.notes[page_number].remove()

    def get_number_of_all_lines(self):
        result = 0
        for k in self.notes.keys():
            result += self.notes[k].get_number_of_lines()
        return result

    def get_number_of_all_characters(self):
        result = 0
        for k in self.notes.keys():
            result += self.notes[k].get_number_of_characters()
        return result

    def get_number_of_all_pages(self):
        return len(self.notes.keys())

    def __str__(self):
        return self.name
