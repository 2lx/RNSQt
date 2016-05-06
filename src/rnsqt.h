#ifndef RNSQT_H
#define RNSQT_H

#include <QtGui/QMainWindow>
#include "ui_rnsqt.h"
#include "rnslogic.h"

class RNSQT : public QMainWindow
{
Q_OBJECT

public:
    RNSQT(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~RNSQT();

private:
    Ui::RNSQTClass ui;

    RNSLogic rnsl;

    bool isResult1Change, isModuli1Change;
    bool isResult2Change, isModuli2Change;
    bool isOperationChange;

    bool isEven;

    QString logCheckResult;

    void checkOperationEnable();
    void checkOperationRight();
    void evenModules( const int which );

private slots:
    void slotCheckModuli();

    void slotModuliCountChanged( const int newCount );
    void slotModuliChanged(const int);

    void slotFirstChanged( );
    void slotFirstResultChanged();

    void slotSecondChanged( );
    void slotSecondResultChanged();

    void slotOperationChanged();

    void slotOperation1CPU();
    void slotOperationGPU();

    void slotAbout();
    void slotSave();
    void slotLoad();
};

#endif // RNSQT_H
