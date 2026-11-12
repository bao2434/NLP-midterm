import pymupdf 
from pathlib import Path

def pdf_split(src: str, lang: str = "ch", skip: int=0):
    print("Source address: ", src)

    path = Path(src)
    if not path.exists():
        raise ValueError("Source file doesn't exist")
    file_name = path.stem
    parent_dir = path.parent
    img_dir = path.parent / lang
    if not img_dir.exists():
        img_dir.mkdir()
    
    print(img_dir)
    
    doc = pymupdf.open(src)
    print("Number of pages:", doc.page_count)
    
    for page in doc:
        if (page.number >= skip):
            pix = page.get_pixmap(
                matrix=None,
                dpi=300,
                colorspace=pymupdf.csRGB,
                clip=None,
                alpha=False,
                annots=True
            )

            img_name = file_name + "." + f"{page.number+1:0{len(str(doc.page_count))}d}" + ".jpg"
            img_path = img_dir / img_name
            pix.save(img_path)
    
    print("Complete splitting")
    
# pdf_split("../data/Linh khởi điểm học trung y: đái nhĩ thể nghiệm đích 12 đường khóa/Linh khởi điểm học trung y: đái nhĩ thể nghiệm trung y đích 12 đường khóa.pdf","ch")
# pdf_split("../data/phuong-te-hoc-cung-trung-y-trung-duoc-cham-cuu-chuyen-nghiep-dung-hua/phuong-te-hoc-cung-trung-y-trung-duoc-cham-cuu-chuyen-nghiep-dung-hua.pdf")
pdf_split("../data/nam_khoa_nu_khoa/nam_khoa_nu_khoa.pdf", "vi", 220)
