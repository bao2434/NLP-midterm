from PIL import Image
import pathlib
import os
def pdf_merge(src_dir: str, file_dst: str, recursive: bool = False): 
    print(src_dir)
    for root, dirs, fls in os.walk(src_dir):
        list = []
        fls.sort()
        for file in fls:
            name = os.path.basename(file)
            base, ext = os.path.splitext(name)
            if (ext != '.jpg'):
                continue

            # parts = file.split("(")
            # if (len(parts) == 2):
            #     continue

            # print(file, len(parts))
            # parts = file.split(".")
            # org = os.path.join(root, file)
            # parts.insert(1, " (q5)")
            # file = (parts[0] + parts[1] + "." + parts[2] + "." + parts[3])
            # new_file = os.path.join(root, file)
            # os.rename(org, new_file)

            # print()
            img = Image.open(os.path.join(root, file))
            if img.mode == 'RGB':
                background = Image.new("RGB", img.size, (255, 0, 0))
                img = img.convert('RGB')
                background.paste(img, mask=img.split()[-1]) 
                list.append(background)
        if (len(list) != 0):
            img = list[0]
            img.save(file_dst, "PDF", save_all=True, append_images=list[1:])
        # print(len(list))
            # img.save(file, "PDF")

# pdf_merge("input.jpg", "output.pdf")
for root, src, files in os.walk("./data/"):
    if (len(files)):
        paths = root.split('/')[:-1]
        paths.append(paths[-1])
        # print(root)
        pdf_merge(root, '/'.join(paths) + ".pdf")
# pdf_merge("./data/Tân san Nam dược thần hiệu thập khoa ứng trị/Tân san Nam dược thần hiệu thập khoa ứng trị (q5)", "tst.pdf")
