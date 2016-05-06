#include <QtGui>
#include "rnsqt.h"

#include "rnslogic.h"
#include "ui_frmAbout.h"

RNSQT::RNSQT(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
    ui.setupUi(this);

    QRegExp rx("[0-9]*,?[0-9]*");
    QValidator * dblval1 = new QRegExpValidator(rx, this);
    QValidator * dblval2 = new QRegExpValidator(rx, this);

    ui.ed1Result->setValidator( dblval1 );
    ui.ed2Result->setValidator( dblval2 );

    isResult1Change = isModuli1Change = isResult2Change = isModuli2Change = isOperationChange = isEven = false;

    slotModuliCountChanged( ui.sbModuliCount->value() );
    slotModuliChanged( 0 );

    QObject::connect( ui.sbModuliCount, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliCountChanged( int ) ) );
    QObject::connect( ui.btnCheckModuli, SIGNAL( clicked() ), this, SLOT( slotCheckModuli() ) );

    QObject::connect( ui.sbRNSm1, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm2, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm3, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm4, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm5, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm6, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm7, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm8, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm9, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );
    QObject::connect( ui.sbRNSm10, SIGNAL( valueChanged( int ) ), this, SLOT( slotModuliChanged( int ) ) );

    QObject::connect( ui.btnCheckModuli, SIGNAL( clicked() ), this, SLOT( slotCheckModuli() ) );

    QObject::connect( ui.sbo1m1, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m2, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m3, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m4, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m5, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m6, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m7, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m8, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged() ) );
    QObject::connect( ui.sbo1m9, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );
    QObject::connect( ui.sbo1m10, SIGNAL( editingFinished() ), this, SLOT( slotFirstChanged( ) ) );

    QObject::connect( ui.ed1Result, SIGNAL( editingFinished() ), this, SLOT( slotFirstResultChanged() ) );

    QObject::connect( ui.sbo2m1, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m2, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m3, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m4, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m5, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m6, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m7, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m8, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m9, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );
    QObject::connect( ui.sbo2m10, SIGNAL( editingFinished() ), this, SLOT( slotSecondChanged( ) ) );

    QObject::connect( ui.ed2Result, SIGNAL( editingFinished() ), this, SLOT( slotSecondResultChanged() ) );

    QObject::connect( ui.rbOpSum, SIGNAL( pressed() ), this, SLOT( slotOperationChanged() ) );
    QObject::connect( ui.rbOpSub, SIGNAL( pressed() ), this, SLOT( slotOperationChanged() ) );
    QObject::connect( ui.rbOpMul, SIGNAL( pressed() ), this, SLOT( slotOperationChanged() ) );
    QObject::connect( ui.rbOpDiv, SIGNAL( pressed() ), this, SLOT( slotOperationChanged() ) );

    QObject::connect( ui.btnOperate1CPU, SIGNAL( clicked() ), this, SLOT( slotOperation1CPU() ) );
    QObject::connect( ui.btnOperateGPU, SIGNAL( clicked() ), this, SLOT( slotOperationGPU() ) );

    QObject::connect( &rnsl, SIGNAL( consoleOut( const QString & ) ), ui.pteOut, SLOT( appendPlainText( const QString & ) ) );

    QObject::connect( ui.btnAbout, SIGNAL( clicked() ), this, SLOT( slotAbout() ) );
    QObject::connect( ui.btnSave, SIGNAL( clicked() ), this, SLOT( slotSave() ) );
    QObject::connect( ui.btnLoad, SIGNAL( clicked() ), this, SLOT( slotLoad() ) );
}

RNSQT::~RNSQT()
{

}

void RNSQT::slotCheckModuli()
{
    ui.pteOut->clear();
    if ( rnsl.checkModuliCoprime() )
    {
        ui.gbFirst->setEnabled( true );
        ui.gbSecond->setEnabled( true );
        ui.gbOperation->setEnabled( true );

        ui.sbo1m1->setMaximum( ui.sbRNSm1->value() - 1 );
        ui.sbo1m2->setMaximum( ui.sbRNSm2->value() - 1 );
        ui.sbo1m3->setMaximum( ui.sbRNSm3->value() - 1 );
        ui.sbo1m4->setMaximum( ui.sbRNSm4->value() - 1 );
        ui.sbo1m5->setMaximum( ui.sbRNSm5->value() - 1 );
        ui.sbo1m6->setMaximum( ui.sbRNSm6->value() - 1 );
        ui.sbo1m7->setMaximum( ui.sbRNSm7->value() - 1 );
        ui.sbo1m8->setMaximum( ui.sbRNSm8->value() - 1 );
        ui.sbo1m9->setMaximum( ui.sbRNSm9->value() - 1 );
        ui.sbo1m10->setMaximum( ui.sbRNSm10->value() - 1 );

        ui.sbo2m1->setMaximum( ui.sbRNSm1->value() - 1 );
        ui.sbo2m2->setMaximum( ui.sbRNSm2->value() - 1 );
        ui.sbo2m3->setMaximum( ui.sbRNSm3->value() - 1 );
        ui.sbo2m4->setMaximum( ui.sbRNSm4->value() - 1 );
        ui.sbo2m5->setMaximum( ui.sbRNSm5->value() - 1 );
        ui.sbo2m6->setMaximum( ui.sbRNSm6->value() - 1 );
        ui.sbo2m7->setMaximum( ui.sbRNSm7->value() - 1 );
        ui.sbo2m8->setMaximum( ui.sbRNSm8->value() - 1 );
        ui.sbo2m9->setMaximum( ui.sbRNSm9->value() - 1 );
        ui.sbo2m10->setMaximum( ui.sbRNSm10->value() - 1 );

        UINT64 maxV = rnsl.getMaxRNSValue();

        ui.edMaxValue->setText( QString::number( maxV ) );

        slotFirstResultChanged();
        slotSecondResultChanged();
    }
}

//-------------------------------------------------------------------------------------------

void RNSQT::slotModuliCountChanged( const int newCount )
{
    ui.gbFirst->setEnabled( false );
    ui.gbSecond->setEnabled( false );
    ui.gbOperation->setEnabled( false );
    ui.gbTotalResult->setEnabled( false );

    ui.sbRNSm3->setEnabled( newCount >= 3 );
    ui.sbRNSm4->setEnabled( newCount >= 4 );
    ui.sbRNSm5->setEnabled( newCount >= 5 );
    ui.sbRNSm6->setEnabled( newCount >= 6 );
    ui.sbRNSm7->setEnabled( newCount >= 7 );
    ui.sbRNSm8->setEnabled( newCount >= 8 );
    ui.sbRNSm9->setEnabled( newCount >= 9 );
    ui.sbRNSm10->setEnabled( newCount >= 10 );

    ui.sbo1m3->setEnabled( newCount >= 3 );
    ui.sbo1m4->setEnabled( newCount >= 4 );
    ui.sbo1m5->setEnabled( newCount >= 5 );
    ui.sbo1m6->setEnabled( newCount >= 6 );
    ui.sbo1m7->setEnabled( newCount >= 7 );
    ui.sbo1m8->setEnabled( newCount >= 8 );
    ui.sbo1m9->setEnabled( newCount >= 9 );
    ui.sbo1m10->setEnabled( newCount >= 10 );

    ui.sbo2m3->setEnabled( newCount >= 3 );
    ui.sbo2m4->setEnabled( newCount >= 4 );
    ui.sbo2m5->setEnabled( newCount >= 5 );
    ui.sbo2m6->setEnabled( newCount >= 6 );
    ui.sbo2m7->setEnabled( newCount >= 7 );
    ui.sbo2m8->setEnabled( newCount >= 8 );
    ui.sbo2m9->setEnabled( newCount >= 9 );
    ui.sbo2m10->setEnabled( newCount >= 10 );

    ui.sbRm3->setEnabled( newCount >= 3 );
    ui.sbRm4->setEnabled( newCount >= 4 );
    ui.sbRm5->setEnabled( newCount >= 5 );
    ui.sbRm6->setEnabled( newCount >= 6 );
    ui.sbRm7->setEnabled( newCount >= 7 );
    ui.sbRm8->setEnabled( newCount >= 8 );
    ui.sbRm9->setEnabled( newCount >= 9 );
    ui.sbRm10->setEnabled( newCount >= 10 );

    rnsl.setModuliCount( newCount );
}

void RNSQT::slotModuliChanged( const int )
{
    ui.gbFirst->setEnabled( false );
    ui.gbSecond->setEnabled( false );
    ui.gbOperation->setEnabled( false );
    ui.gbTotalResult->setEnabled( false );

    rnsl.setModuliValue( 0, ui.sbRNSm1->value() );
    rnsl.setModuliValue( 1, ui.sbRNSm2->value() );
    rnsl.setModuliValue( 2, ui.sbRNSm3->value() );
    rnsl.setModuliValue( 3, ui.sbRNSm4->value() );
    rnsl.setModuliValue( 4, ui.sbRNSm5->value() );
    rnsl.setModuliValue( 5, ui.sbRNSm6->value() );
    rnsl.setModuliValue( 6, ui.sbRNSm7->value() );
    rnsl.setModuliValue( 7, ui.sbRNSm8->value() );
    rnsl.setModuliValue( 8, ui.sbRNSm9->value() );
    rnsl.setModuliValue( 9, ui.sbRNSm10->value() );
}

//-------------------------------------------------------------------------------------------

void RNSQT::checkOperationEnable()
{
    ui.pteOut->clear();

    isOperationChange = true;

    logCheckResult = "Восстановление возможно.";

    // проверка возможности деления
    if ( ui.rbOpDiv->isChecked() )
    {
        if ( ! ui.ed1Result->text().replace( ",", "" ).toLong() )
            logCheckResult = "Восстановление невозможно. Деление нуля запрещено";
        else if ( ! ui.ed2Result->text().replace( ",", "" ).toLong() )
            logCheckResult = "Восстановление невозможно. Деление на ноль запрещено";
        else if ( ( ( ui.ed1Result->text().replace( ",", "" ).toLong() % ui.ed2Result->text().replace( ",", "" ).toLong() ) != 0 ) || ( ui.ed1Result->text().replace( ",", "" ).toLong() < ui.ed2Result->text().replace( ",", "" ).toLong() ) )
            logCheckResult = "Восстановление невозможно. Деление с остатком запрещено";
        else if ( rnsl.getSecondDividedResult( ui.ed2Result->text().replace( ",", "" ).toLong() ) )
            logCheckResult = "Восстановление невозможно. Деление на делитель, кратный любому остатку, запрещено";
    }

    // проверка возможности сложения
    if ( ui.rbOpSum->isChecked() )
    {
        if ( ( ui.ed1Result->text().replace( ",", "" ).toLong() + ui.ed2Result->text().replace( ",", "" ).toLong() ) > rnsl.getMaxRNSValue() )
            logCheckResult = "Сложение операндов, сумма которых больше максимума СОК, запрещено";
    }

    // проверка возможности разности
    if ( ui.rbOpSub->isChecked() )
    {
        if ( ui.ed1Result->text().replace( ",", "" ).toLong() < ui.ed2Result->text().replace( ",", "" ).toLong() )
        {
            isOperationChange = true;
            const QString iTemp = ui.ed2Result->text();
            ui.ed2Result->setText( ui.ed1Result->text() );
            ui.ed1Result->setText( iTemp );
            isOperationChange = false;
        }
    }

    // проверка возможности умножения
    if ( ui.rbOpMul->isChecked() )
    {
        if ( ( ui.ed1Result->text().replace( ",", "" ).toLong() * ui.ed2Result->text().replace( ",", "" ).toLong() ) > rnsl.getMaxRNSValue() )
            logCheckResult = "Умножение операндов, произведение которых больше максимума СОК, запрещено";
    }

    ui.pteOut->appendPlainText( "" );

    isOperationChange = false;
}

//-------------------------------------------------------------------------------------------

void RNSQT::evenModules( const int which )
{
    while( ui.sbo2Factor->value() < ui.sbo1Factor->value() )
    {
        if ( ui.ed2Result->text().at( ui.ed2Result->text().length() - 1 ) == '0' )
        {
            if ( which == 2 )
                isModuli2Change = true;

            ui.ed2Result->setText( ui.ed2Result->text().left( ui.ed2Result->text().length() - 1 ) );
            ui.sbo2Factor->setValue( ui.sbo2Factor->value() + 1 );
            if ( ui.sbo2Factor->value() == 0 )
                ui.ed2Result->setText( ui.ed2Result->text().left( ui.ed2Result->text().length() - 1 ) );

            if ( which != 2 )
                slotSecondResultChanged();

            if ( which == 2 )
                isModuli2Change = false;
        }
        else
        {
            if ( which == 1 )
                isModuli1Change = true;

            if ( ui.sbo1Factor->value() == 0 )
                ui.ed1Result->setText( ui.ed1Result->text().replace( "," , "" ) + ",0" );
            else ui.ed1Result->setText( ui.ed1Result->text() + "0" );
            ui.sbo1Factor->setValue( ui.sbo1Factor->value() - 1 );

            if ( ui.ed1Result->text() == ",0" )
                ui.ed1Result->setText( "0,0" );

            if ( which != 1 )
                slotFirstResultChanged();

            if ( which == 1 )
                isModuli1Change = false;
        }
    }

    while( ui.sbo1Factor->value() < ui.sbo2Factor->value() )
    {
        if ( ui.ed1Result->text().at( ui.ed1Result->text().length() - 1 ) == '0' )
        {
            if ( which == 1 )
                isModuli1Change = true;

            ui.ed1Result->setText( ui.ed1Result->text().left( ui.ed1Result->text().length() - 1 ) );
            ui.sbo1Factor->setValue( ui.sbo1Factor->value() + 1 );
            if ( ui.sbo1Factor->value() == 0 )
                ui.ed1Result->setText( ui.ed1Result->text().left( ui.ed1Result->text().length() - 1 ) );

            if ( which != 1 )
                slotFirstResultChanged();

            if ( which == 1 )
                isModuli1Change = false;
        }
        else
        {
            if ( which == 2 )
                isModuli2Change = true;

            if ( ui.sbo2Factor->value() == 0 )
                ui.ed2Result->setText( ui.ed2Result->text().replace( "," , "" ) + ",0" );
            else ui.ed2Result->setText( ui.ed2Result->text() + "0" );
            ui.sbo2Factor->setValue( ui.sbo2Factor->value() - 1 );

            if ( ui.ed2Result->text() == ",0" )
                ui.ed2Result->setText( "0,0" );

            if ( which != 2 )
                slotSecondResultChanged();

            if ( which == 2 )
                isModuli2Change = false;
        }
    }

    isEven = true;
    slotFirstResultChanged();
    slotSecondResultChanged();
    isEven = false;
}

//-------------------------------------------------------------------------------------------

void RNSQT::slotFirstChanged( )
{
    if ( isResult1Change ) return;

    ui.gbTotalResult->setEnabled( false );

    rnsl.setFirstOpBitValue( 0, ui.sbo1m1->value() );
    rnsl.setFirstOpBitValue( 1, ui.sbo1m2->value() );
    rnsl.setFirstOpBitValue( 2, ui.sbo1m3->value() );
    rnsl.setFirstOpBitValue( 3, ui.sbo1m4->value() );
    rnsl.setFirstOpBitValue( 4, ui.sbo1m5->value() );
    rnsl.setFirstOpBitValue( 5, ui.sbo1m6->value() );
    rnsl.setFirstOpBitValue( 6, ui.sbo1m7->value() );
    rnsl.setFirstOpBitValue( 7, ui.sbo1m8->value() );
    rnsl.setFirstOpBitValue( 8, ui.sbo1m9->value() );
    rnsl.setFirstOpBitValue( 9, ui.sbo1m10->value() );

    isModuli1Change = true;

    QString resStr = QString::number( rnsl.getFirstResult() );
    if ( ui.sbo1Factor->value() )
        resStr.insert( resStr.length() + ui.sbo1Factor->value(), "," );
    ui.ed1Result->setText( resStr );

    isModuli1Change = false;

    if ( !isOperationChange )
        checkOperationEnable();

    ui.gbTotalResult->setEnabled( false );
}

void RNSQT::slotFirstResultChanged()
{
    if ( isModuli1Change ) return;

    ui.gbTotalResult->setEnabled( false );

    QString strRes = ui.ed1Result->text();
    rnsl.setFirstResult( strRes.replace( "," , "" ).toLong() );
    if ( ui.ed1Result->text().indexOf( ",", 0 ) > -1 )
        ui.sbo1Factor->setValue( - ui.ed1Result->text().length() + ui.ed1Result->text().indexOf( ",", 0 ) + 1 );
    else ui.sbo1Factor->setValue( 0 );

    if ( !isEven )
        evenModules( 1 );

    if ( !isOperationChange )
        checkOperationEnable();

    isResult1Change = true;

    ui.sbo1m1->setValue( rnsl.getFirstOpBitValue( 0 ) );
    ui.sbo1m2->setValue( rnsl.getFirstOpBitValue( 1 ) );
    ui.sbo1m3->setValue( rnsl.getFirstOpBitValue( 2 ) );
    ui.sbo1m4->setValue( rnsl.getFirstOpBitValue( 3 ) );
    ui.sbo1m5->setValue( rnsl.getFirstOpBitValue( 4 ) );
    ui.sbo1m6->setValue( rnsl.getFirstOpBitValue( 5 ) );
    ui.sbo1m7->setValue( rnsl.getFirstOpBitValue( 6 ) );
    ui.sbo1m8->setValue( rnsl.getFirstOpBitValue( 7 ) );
    ui.sbo1m9->setValue( rnsl.getFirstOpBitValue( 8 ) );
    ui.sbo1m10->setValue( rnsl.getFirstOpBitValue( 9 ) );

    isResult1Change = false;

    ui.gbTotalResult->setEnabled( false );
}

//-------------------------------------------------------------------------------------------

void RNSQT::slotSecondChanged(  )
{
    if ( isResult2Change ) return;

    ui.gbTotalResult->setEnabled( false );

    rnsl.setSecondOpBitValue( 0, ui.sbo2m1->value() );
    rnsl.setSecondOpBitValue( 1, ui.sbo2m2->value() );
    rnsl.setSecondOpBitValue( 2, ui.sbo2m3->value() );
    rnsl.setSecondOpBitValue( 3, ui.sbo2m4->value() );
    rnsl.setSecondOpBitValue( 4, ui.sbo2m5->value() );
    rnsl.setSecondOpBitValue( 5, ui.sbo2m6->value() );
    rnsl.setSecondOpBitValue( 6, ui.sbo2m7->value() );
    rnsl.setSecondOpBitValue( 7, ui.sbo2m8->value() );
    rnsl.setSecondOpBitValue( 8, ui.sbo2m9->value() );
    rnsl.setSecondOpBitValue( 9, ui.sbo2m10->value() );

    isModuli2Change = true;

    QString resStr = QString::number( rnsl.getSecondResult() );
    if ( ui.sbo2Factor->value() )
        resStr.insert( resStr.length() + ui.sbo2Factor->value(), "," );
    ui.ed2Result->setText( resStr );

    isModuli2Change = false;

    if ( !isOperationChange )
        checkOperationEnable();

    ui.gbTotalResult->setEnabled( false );
}

void RNSQT::slotSecondResultChanged()
{
    if ( isModuli2Change ) return;

    ui.gbTotalResult->setEnabled( false );

    QString strRes = ui.ed2Result->text();
    rnsl.setSecondResult( strRes.replace( "," , "" ).toLong() );
    if (  ui.ed2Result->text().indexOf( ",", 0 ) > -1 )
        ui.sbo2Factor->setValue( -  ui.ed2Result->text().length() +  ui.ed2Result->text().indexOf( ",", 0 ) + 1 );
    else ui.sbo2Factor->setValue( 0 );

    if ( !isEven )
        evenModules( 2 );

    if ( !isOperationChange )
        checkOperationEnable();

    isResult2Change = true;
    ui.sbo2m1->setValue( rnsl.getSecondOpBitValue( 0 ) );
    ui.sbo2m2->setValue( rnsl.getSecondOpBitValue( 1 ) );
    ui.sbo2m3->setValue( rnsl.getSecondOpBitValue( 2 ) );
    ui.sbo2m4->setValue( rnsl.getSecondOpBitValue( 3 ) );
    ui.sbo2m5->setValue( rnsl.getSecondOpBitValue( 4 ) );
    ui.sbo2m6->setValue( rnsl.getSecondOpBitValue( 5 ) );
    ui.sbo2m7->setValue( rnsl.getSecondOpBitValue( 6 ) );
    ui.sbo2m8->setValue( rnsl.getSecondOpBitValue( 7 ) );
    ui.sbo2m9->setValue( rnsl.getSecondOpBitValue( 8 ) );
    ui.sbo2m10->setValue( rnsl.getSecondOpBitValue( 9 ) );
    isResult2Change = false;

    ui.gbTotalResult->setEnabled( false );
}

void RNSQT::slotOperationChanged()
{
    if ( isOperationChange ) return;

    checkOperationEnable();
    ui.gbTotalResult->setEnabled( false );
}

//-------------------------------------------------------------------------------------------

void RNSQT::checkOperationRight()
{
    ui.pteOut->appendPlainText( logCheckResult );
    if ( ui.rbOpSum->isChecked() )
    {
        if ( ( ui.ed1Result->text().replace( ",", "" ).toLong() + ui.ed2Result->text().replace( ",", "" ).toLong() ) == ui.edTotalResult->text().replace( ",", "" ).toLong() )
            ui.pteOut->appendPlainText( "Сумма вычислена правильно" );
        else ui.pteOut->appendPlainText( "Сумма вычислена НЕ правильно" );

        ui.sboResultFactor->setValue( ui.sbo1Factor->value() );
    }
    else if ( ui.rbOpSub->isChecked() )
    {
        if ( ( ui.ed1Result->text().replace( ",", "" ).toLong() - ui.ed2Result->text().replace( ",", "" ).toLong() ) == ui.edTotalResult->text().replace( ",", "" ).toLong() )
            ui.pteOut->appendPlainText( "Разность вычислена правильно" );
        else ui.pteOut->appendPlainText( "Разность вычислена НЕ правильно" );

        ui.sboResultFactor->setValue( ui.sbo1Factor->value() );
    }
    else if ( ui.rbOpMul->isChecked() )
    {
        if ( ( ui.ed1Result->text().replace( ",", "" ).toLong() * ui.ed2Result->text().replace( ",", "" ).toLong() ) == ui.edTotalResult->text().replace( ",", "" ).toLong() )
            ui.pteOut->appendPlainText( "Произведение вычислено правильно" );
        else ui.pteOut->appendPlainText( "Произведение вычислено НЕ правильно" );

        ui.sboResultFactor->setValue( 2 * ui.sbo1Factor->value() );
    }
    else if ( ui.rbOpDiv->isChecked() )
    {

        if ( ( ui.ed2Result->text().replace( ",", "" ).toLong() ) &&
            ( ( ui.ed1Result->text().replace( ",", "" ).toLong() / ui.ed2Result->text().replace( ",", "" ).toLong() ) == ui.edTotalResult->text().replace( ",", "" ).toLong() ) )
            ui.pteOut->appendPlainText( "Частное вычислено правильно" );
        else ui.pteOut->appendPlainText( "Частное вычислено НЕ правильно" );

        ui.sboResultFactor->setValue( 0 );
    };

    QString resStr = ui.edTotalResult->text();
    if ( ui.sboResultFactor->value() )
        resStr.insert( resStr.length() + ui.sboResultFactor->value(), "," );
    ui.edTotalResult->setText( resStr );
}

//-------------------------------------------------------------------------------------------

void RNSQT::slotOperation1CPU()
{
    ui.pteOut->clear();

    if ( ui.rbOpSum->isChecked() )
        rnsl.SumRNS( 1 );
    else if ( ui.rbOpSub->isChecked() )
        rnsl.SumRNS( 2 );
    else if ( ui.rbOpMul->isChecked() )
        rnsl.SumRNS( 3 );
    else if ( ui.rbOpDiv->isChecked() )
        rnsl.SumRNS( 4 );

    ui.sbRm1->setValue( rnsl.getResultBitValue( 0 ) );
    ui.sbRm2->setValue( rnsl.getResultBitValue( 1 ) );
    ui.sbRm3->setValue( rnsl.getResultBitValue( 2 ) );
    ui.sbRm4->setValue( rnsl.getResultBitValue( 3 ) );
    ui.sbRm5->setValue( rnsl.getResultBitValue( 4 ) );
    ui.sbRm6->setValue( rnsl.getResultBitValue( 5 ) );
    ui.sbRm7->setValue( rnsl.getResultBitValue( 6 ) );
    ui.sbRm8->setValue( rnsl.getResultBitValue( 7 ) );
    ui.sbRm9->setValue( rnsl.getResultBitValue( 8 ) );
    ui.sbRm10->setValue( rnsl.getResultBitValue( 9 ) );

    ui.edTotalResult->setText( QString::number( rnsl.getTotalResult() ) );

    ui.gbTotalResult->setEnabled( true );
    checkOperationRight();
}

//-------------------------------------------------------------------------------------------

void RNSQT::slotOperationGPU()
{
    ui.pteOut->clear();

    if ( ui.rbOpSum->isChecked() )
        rnsl.SumRNSCUDA( 1 );
    else if ( ui.rbOpSub->isChecked() )
        rnsl.SumRNSCUDA( 2 );
    else if ( ui.rbOpMul->isChecked() )
        rnsl.SumRNSCUDA( 3 );
    else if ( ui.rbOpDiv->isChecked() )
        rnsl.SumRNSCUDA( 4 );

    ui.sbRm1->setValue( rnsl.getResultBitValue( 0 ) );
    ui.sbRm2->setValue( rnsl.getResultBitValue( 1 ) );
    ui.sbRm3->setValue( rnsl.getResultBitValue( 2 ) );
    ui.sbRm4->setValue( rnsl.getResultBitValue( 3 ) );
    ui.sbRm5->setValue( rnsl.getResultBitValue( 4 ) );
    ui.sbRm6->setValue( rnsl.getResultBitValue( 5 ) );
    ui.sbRm7->setValue( rnsl.getResultBitValue( 6 ) );
    ui.sbRm8->setValue( rnsl.getResultBitValue( 7 ) );
    ui.sbRm9->setValue( rnsl.getResultBitValue( 8 ) );
    ui.sbRm10->setValue( rnsl.getResultBitValue( 9 ) );

    ui.edTotalResult->setText( QString::number( rnsl.getTotalResult() ) );

    ui.gbTotalResult->setEnabled( true );
    checkOperationRight();
}

void RNSQT::slotAbout()
{
    QDialog * dlg = new QDialog( this );

    Ui::dlgAbout uia;
    uia.setupUi( dlg );

    dlg->setWindowModality( Qt::WindowModal );
    dlg->show();
}

void RNSQT::slotSave()
{
    QString settingsFile = QFileDialog::getSaveFileName( this, tr( "Сохранение файла" ), "", tr( "Files (*.sett)" ) );

    QSettings settings( settingsFile, QSettings::IniFormat );

    settings.clear();

    settings.setValue( "RNSm1", ui.sbRNSm1->value() );
    settings.setValue( "RNSm2", ui.sbRNSm2->value() );
    settings.setValue( "RNSm3", ui.sbRNSm3->value() );
    settings.setValue( "RNSm4", ui.sbRNSm4->value() );
    settings.setValue( "RNSm5", ui.sbRNSm5->value() );
    settings.setValue( "RNSm6", ui.sbRNSm6->value() );
    settings.setValue( "RNSm7", ui.sbRNSm7->value() );
    settings.setValue( "RNSm8", ui.sbRNSm8->value() );
    settings.setValue( "RNSm9", ui.sbRNSm9->value() );
    settings.setValue( "RNSm10", ui.sbRNSm10->value() );

    settings.setValue( "valRNSCount", ui.sbModuliCount->value() );

    settings.setValue( "valO1m1", ui.sbo1m1->value() );
    settings.setValue( "valO1m2", ui.sbo1m2->value() );
    settings.setValue( "valO1m3", ui.sbo1m3->value() );
    settings.setValue( "valO1m4", ui.sbo1m4->value() );
    settings.setValue( "valO1m5", ui.sbo1m5->value() );
    settings.setValue( "valO1m6", ui.sbo1m6->value() );
    settings.setValue( "valO1m7", ui.sbo1m7->value() );
    settings.setValue( "valO1m8", ui.sbo1m8->value() );
    settings.setValue( "valO1m9", ui.sbo1m9->value() );
    settings.setValue( "valO1m10", ui.sbo1m10->value() );

    settings.setValue( "valO1Factor", ui.sbo1Factor->value() );

    settings.setValue( "valO2m1", ui.sbo2m1->value() );
    settings.setValue( "valO2m2", ui.sbo2m2->value() );
    settings.setValue( "valO2m3", ui.sbo2m3->value() );
    settings.setValue( "valO2m4", ui.sbo2m4->value() );
    settings.setValue( "valO2m5", ui.sbo2m5->value() );
    settings.setValue( "valO2m6", ui.sbo2m6->value() );
    settings.setValue( "valO2m7", ui.sbo2m7->value() );
    settings.setValue( "valO2m8", ui.sbo2m8->value() );
    settings.setValue( "valO2m9", ui.sbo2m9->value() );
    settings.setValue( "valO2m10", ui.sbo2m10->value() );

    settings.setValue( "valO2Factor", ui.sbo2Factor->value() );

    settings.setValue( "result1", ui.ed1Result->text() );
    settings.setValue( "result2", ui.ed2Result->text() );

    settings.sync();
}

void RNSQT::slotLoad()
{
    QString settingsFile = QFileDialog::getOpenFileName( this, tr( "Открытие файла" ), "", tr( "Files (*.sett)" ) );

    QSettings settings( settingsFile, QSettings::IniFormat );

    ui.sbRNSm1->setValue( settings.value( "RNSm1" ).toInt() );
    ui.sbRNSm2->setValue( settings.value( "RNSm2" ).toInt() );
    ui.sbRNSm3->setValue( settings.value( "RNSm3" ).toInt() );
    ui.sbRNSm4->setValue( settings.value( "RNSm4" ).toInt() );
    ui.sbRNSm5->setValue( settings.value( "RNSm5" ).toInt() );
    ui.sbRNSm6->setValue( settings.value( "RNSm6" ).toInt() );
    ui.sbRNSm7->setValue( settings.value( "RNSm7" ).toInt() );
    ui.sbRNSm8->setValue( settings.value( "RNSm8" ).toInt() );
    ui.sbRNSm9->setValue( settings.value( "RNSm9" ).toInt() );
    ui.sbRNSm10->setValue( settings.value( "RNSm10" ).toInt() );

    ui.sbModuliCount->setValue( settings.value( "valRNSCount" ).toInt() );

    ui.sbo1m1->setValue( settings.value( "valO1m1" ).toInt() );
    ui.sbo1m2->setValue( settings.value( "valO1m2" ).toInt() );
    ui.sbo1m3->setValue( settings.value( "valO1m3" ).toInt() );
    ui.sbo1m4->setValue( settings.value( "valO1m4" ).toInt() );
    ui.sbo1m5->setValue( settings.value( "valO1m5" ).toInt() );
    ui.sbo1m6->setValue( settings.value( "valO1m6" ).toInt() );
    ui.sbo1m7->setValue( settings.value( "valO1m7" ).toInt() );
    ui.sbo1m8->setValue( settings.value( "valO1m8" ).toInt() );
    ui.sbo1m9->setValue( settings.value( "valO1m9" ).toInt() );
    ui.sbo1m10->setValue( settings.value( "valO1m10" ).toInt() );

    ui.sbo1Factor->setValue( settings.value( "valO1Factor" ).toInt() );

    ui.sbo2m1->setValue( settings.value( "valO2m1" ).toInt() );
    ui.sbo2m2->setValue( settings.value( "valO2m2" ).toInt() );
    ui.sbo2m3->setValue( settings.value( "valO2m3" ).toInt() );
    ui.sbo2m4->setValue( settings.value( "valO2m4" ).toInt() );
    ui.sbo2m5->setValue( settings.value( "valO2m5" ).toInt() );
    ui.sbo2m6->setValue( settings.value( "valO2m6" ).toInt() );
    ui.sbo2m7->setValue( settings.value( "valO2m7" ).toInt() );
    ui.sbo2m8->setValue( settings.value( "valO2m8" ).toInt() );
    ui.sbo2m9->setValue( settings.value( "valO2m9" ).toInt() );
    ui.sbo2m10->setValue( settings.value( "valO2m10" ).toInt() );

    ui.sbo2Factor->setValue( settings.value( "valO2Factor" ).toInt() );

    ui.ed1Result->setText( settings.value( "result1" ).toString() );
    ui.ed2Result->setText( settings.value( "result2" ).toString() );
}
