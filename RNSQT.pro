TEMPLATE = app
TARGET = RNSQt

cudatarget.target = kernelcode
cudatarget.commands = nvcc -c ./src/kernelcode.cu -o ./obj/kernelcode.o
QMAKE_EXTRA_TARGETS += cudatarget
PRE_TARGETDEPS += kernelcode

win:DESTDIR = ./
!win:DESTDIR = ./
QT += core gui
CONFIG += release
DEFINES += QT_DLL
INCLUDEPATH += ./GeneratedFiles \
    ./src
LIBS += ./obj/kernelcode.o -lcudart
DEPENDPATH += .
MOC_DIR += ./GeneratedFiles
OBJECTS_DIR += obj
UI_DIR += ./GeneratedFiles
RCC_DIR += ./GeneratedFiles
include(RNSQT.pri)
