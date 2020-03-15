import xlsxwriter

workbook=xlsxwriter.Workbook('hello.xlsx')

worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Hello world')

worksheet.write(1,1,'guoshun')

workbook.close()