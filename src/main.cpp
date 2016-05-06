#include <QtGui>
#include "rnsqt.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{	
	QApplication a(argc, argv);

    QTextCodec* codec = QTextCodec::codecForName( "UTF-8" );
    QTextCodec::setCodecForCStrings(codec);

	RNSQT w;
    QDesktopWidget wid;

    const int hsWidth = wid.screen()->width() / 2;
    const int hsHeight = wid.screen()->height() / 2;

    w.move( hsWidth - (w.width() / 2), hsHeight - (w.height() / 2) );
    w.show();

	return a.exec();
}
