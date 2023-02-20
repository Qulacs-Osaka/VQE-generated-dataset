OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.1415222704004657) q[0];
rz(-1.8535044976121826) q[0];
ry(3.1415854742521243) q[1];
rz(-1.9988490807304768) q[1];
ry(-3.1399727983519248) q[2];
rz(0.7146129799728831) q[2];
ry(-3.141497650893199) q[3];
rz(0.9166329952715246) q[3];
ry(4.577390150917182e-05) q[4];
rz(1.7639630277986686) q[4];
ry(-1.5440995639119497) q[5];
rz(-0.03283668746612012) q[5];
ry(5.571988968000596e-07) q[6];
rz(-0.35905517520822444) q[6];
ry(3.092938873688131) q[7];
rz(-2.356409951307282) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5710143415985487) q[0];
rz(-0.0006356759935948665) q[0];
ry(1.570726089568116) q[1];
rz(-0.010110770347377158) q[1];
ry(-8.43759450473879e-05) q[2];
rz(1.8876944080826679) q[2];
ry(-3.141569283225298) q[3];
rz(0.6168239517667089) q[3];
ry(-3.1410569392340904) q[4];
rz(-0.6721748568501393) q[4];
ry(-1.511212035899775) q[5];
rz(0.6819668132224397) q[5];
ry(1.5707913644602816) q[6];
rz(-3.141583992313163) q[6];
ry(1.6163261285951094) q[7];
rz(0.17414317823274492) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.081851816960319) q[0];
rz(-0.3309414335250169) q[0];
ry(-0.021779109746061592) q[1];
rz(-3.0597256869637337) q[1];
ry(-1.5818419484694894) q[2];
rz(-1.6973277979099644) q[2];
ry(1.5470408215271174) q[3];
rz(-0.8585665257644975) q[3];
ry(0.06469384085465624) q[4];
rz(0.29004415583914644) q[4];
ry(-3.141525544357905) q[5];
rz(2.450308880174114) q[5];
ry(1.6522524005062282) q[6];
rz(2.829497034789892) q[6];
ry(-3.1415356553115887) q[7];
rz(-0.6844435092989061) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.02395131984500853) q[0];
rz(0.33039604812134193) q[0];
ry(3.0781248225461137) q[1];
rz(-3.0696905378414785) q[1];
ry(0.7362950963673702) q[2];
rz(1.4120239872721692) q[2];
ry(1.333578907436217) q[3];
rz(-1.9101692262439727) q[3];
ry(0.7194902522158798) q[4];
rz(3.1095257004417682) q[4];
ry(3.109915751895262) q[5];
rz(0.4568726880817957) q[5];
ry(3.0027655013925493e-06) q[6];
rz(-1.397993447086737) q[6];
ry(3.138609803695594) q[7];
rz(-1.3431558493438507) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5705846518541389) q[0];
rz(3.1415522375936624) q[0];
ry(1.5708095228551917) q[1];
rz(-4.5364699089667226e-05) q[1];
ry(-1.5699375940459375) q[2];
rz(-1.568846171293801) q[2];
ry(-3.134974833162171) q[3];
rz(-0.2582392053870475) q[3];
ry(-3.1320516414973536) q[4];
rz(0.8347448779148675) q[4];
ry(3.1387179965255423) q[5];
rz(-2.0961160664778484) q[5];
ry(3.7924987905668672e-06) q[6];
rz(0.9563968185097507) q[6];
ry(0.00020093795465569997) q[7];
rz(2.0545313742659386) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5708040782839623) q[0];
rz(-3.129148861348144) q[0];
ry(1.5707993963383302) q[1];
rz(-3.1401245478338033) q[1];
ry(2.2446856371541934) q[2];
rz(0.9420618149859434) q[2];
ry(0.6527631355879138) q[3];
rz(-0.9512198605636862) q[3];
ry(0.9799456845053802) q[4];
rz(3.088753321430192) q[4];
ry(-1.4725490477879504) q[5];
rz(-0.8043803509515229) q[5];
ry(2.8985350964738643e-05) q[6];
rz(0.7536939507002758) q[6];
ry(-1.556526206014425) q[7];
rz(-1.5783937964339128) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5707867928326504) q[0];
rz(2.246329556190674) q[0];
ry(1.570792591443921) q[1];
rz(-0.41612318335137416) q[1];
ry(3.141592476276902) q[2];
rz(-0.6307389899392852) q[2];
ry(3.1415916469754226) q[3];
rz(2.3904831847378705) q[3];
ry(3.1415912151357763) q[4];
rz(0.342027359045395) q[4];
ry(-4.758007463578906e-07) q[5];
rz(-0.7668854534576433) q[5];
ry(1.570928824140477) q[6];
rz(1.5700275048204064) q[6];
ry(-3.5477271174612184e-05) q[7];
rz(-0.1362319608452947) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.639906717988601) q[0];
rz(-2.153253827939876) q[0];
ry(0.5048145758458685) q[1];
rz(2.353922719552058) q[1];
ry(-1.5704560445346634) q[2];
rz(1.2289522612046797) q[2];
ry(-0.0018600047550573545) q[3];
rz(-1.8877266730512474) q[3];
ry(-3.1108510538744976) q[4];
rz(-0.6951482598753929) q[4];
ry(-1.5740513464583064) q[5];
rz(2.4297546133578556) q[5];
ry(1.571085370672284) q[6];
rz(-2.1418892909044143) q[6];
ry(0.001764765181780119) q[7];
rz(2.7415923226652024) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.23443584658278271) q[0];
rz(0.18155607901602) q[0];
ry(-1.794168104305423) q[1];
rz(0.06290618797697523) q[1];
ry(-5.4145545647608345e-05) q[2];
rz(-1.2266665562322612) q[2];
ry(-0.0007108637302861709) q[3];
rz(-1.3543178526493334) q[3];
ry(-8.722051345788628e-05) q[4];
rz(-1.846538047430559) q[4];
ry(5.313015424501799e-05) q[5];
rz(3.106672229055593) q[5];
ry(3.1415840707234794) q[6];
rz(1.410581684782219) q[6];
ry(-3.1415880578725646) q[7];
rz(2.6568900616158078) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5070849119005194) q[0];
rz(1.9269486506566398) q[0];
ry(1.3752918020847038) q[1];
rz(-0.3453199707586025) q[1];
ry(1.5710652117641626) q[2];
rz(-1.6604662605428278) q[2];
ry(0.0035925291325219046) q[3];
rz(-0.6541761157776779) q[3];
ry(-0.0027982834120579827) q[4];
rz(2.0898349405093732) q[4];
ry(3.141549627430224) q[5];
rz(1.3849388384284511) q[5];
ry(-3.0978705801711754e-05) q[6];
rz(-0.5444082969067062) q[6];
ry(2.204829890827797e-05) q[7];
rz(1.6444689061626232) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.4958250893286857) q[0];
rz(1.8977218357387335) q[0];
ry(-1.6457528678973965) q[1];
rz(-1.243887260095045) q[1];
ry(-0.9930884418928893) q[2];
rz(-2.7657021476457335) q[2];
ry(3.000028648343102) q[3];
rz(-0.22553791086948363) q[3];
ry(0.1351124343479988) q[4];
rz(2.8852271930358144) q[4];
ry(3.107072499931588) q[5];
rz(-0.9842908803438712) q[5];
ry(1.6155359884747176) q[6];
rz(1.5943580485712041) q[6];
ry(-1.526398514131732) q[7];
rz(1.5943958776698113) q[7];