OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.774439647738843) q[0];
rz(-1.2460773415586033) q[0];
ry(-2.8977974017120114) q[1];
rz(-0.45242478063347513) q[1];
ry(-1.5024615138239448) q[2];
rz(1.5169861587593698) q[2];
ry(-1.8103646108677551) q[3];
rz(-2.497950475820672) q[3];
ry(-0.00014041026866884465) q[4];
rz(-2.051548290913334) q[4];
ry(1.570183209739038) q[5];
rz(0.8799427819943517) q[5];
ry(1.571170040503821) q[6];
rz(0.0015011330113133708) q[6];
ry(-1.5726517834092997) q[7];
rz(-1.5233438552556287) q[7];
ry(6.463758764407856e-05) q[8];
rz(-1.4949892993118752) q[8];
ry(8.51181435272963e-05) q[9];
rz(-0.17757015058648631) q[9];
ry(0.002494369726576906) q[10];
rz(-3.1165578509744885) q[10];
ry(-1.7450875087905755) q[11];
rz(2.0189280511745165) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5149267699069604) q[0];
rz(1.1225014030053435) q[0];
ry(1.947795375722002) q[1];
rz(-1.3881835534598146) q[1];
ry(-1.1082906714531697) q[2];
rz(1.5002692905266526) q[2];
ry(-3.1412616616251956) q[3];
rz(-1.8208845969244107) q[3];
ry(3.0873343492051233) q[4];
rz(0.06425118738457414) q[4];
ry(-3.1413755379022166) q[5];
rz(-1.9203523139306005) q[5];
ry(-1.5706962722124316) q[6];
rz(-2.4738080172973147) q[6];
ry(1.9271923335964651) q[7];
rz(1.196938732284595) q[7];
ry(7.327865906958664e-05) q[8];
rz(0.24446054481128432) q[8];
ry(1.5707186836631815) q[9];
rz(0.6907427562323818) q[9];
ry(-3.0893820647297052) q[10];
rz(-3.090854146000272) q[10];
ry(-0.8611499399394632) q[11];
rz(2.7265931652544944) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.007126602961388784) q[0];
rz(0.34564025342847415) q[0];
ry(-1.4909864736530725) q[1];
rz(-3.132946388043322) q[1];
ry(-1.5194734395731695) q[2];
rz(-2.013205879422798) q[2];
ry(0.6531263553808088) q[3];
rz(-0.9833988495012091) q[3];
ry(1.572017009769695) q[4];
rz(1.3070090817443365) q[4];
ry(0.0046708681142870745) q[5];
rz(-0.26854603620868084) q[5];
ry(1.6228838451004108) q[6];
rz(-0.0015104186942666096) q[6];
ry(0.6031368152237704) q[7];
rz(-0.8070754020267036) q[7];
ry(1.5712527623486165) q[8];
rz(-1.5725589714221142) q[8];
ry(0.00024287680364842856) q[9];
rz(-0.6900979512455082) q[9];
ry(3.1411856424627995) q[10];
rz(1.1246739185961472) q[10];
ry(-3.141361038746674) q[11];
rz(3.1234216016458802) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.6122659283200473) q[0];
rz(1.36517223953663) q[0];
ry(-1.3212148521612894) q[1];
rz(1.6352063388817897) q[1];
ry(0.0018078359527247275) q[2];
rz(-0.9056475295933577) q[2];
ry(3.1415709528021227) q[3];
rz(-2.409674058885514) q[3];
ry(5.57090049464916e-06) q[4];
rz(-1.8371940040842407) q[4];
ry(-1.5716012789961122) q[5];
rz(0.00010476171438066257) q[5];
ry(-2.6217770427793328) q[6];
rz(-2.5620196031202185) q[6];
ry(1.6069650529810708) q[7];
rz(-1.6090576405943224) q[7];
ry(0.7890026752697001) q[8];
rz(1.5719860591790908) q[8];
ry(1.5704357249339251) q[9];
rz(0.9390093730174158) q[9];
ry(3.1369086435314166) q[10];
rz(1.121263344646534) q[10];
ry(-0.12449590285278257) q[11];
rz(-2.0484477640822196) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.6309710743189538) q[0];
rz(1.09990509172516) q[0];
ry(-1.553130223297983) q[1];
rz(1.9004144472438824) q[1];
ry(-0.005795482000479524) q[2];
rz(3.0382227390272654) q[2];
ry(1.5689548662863588) q[3];
rz(-0.00127202741263539) q[3];
ry(0.0004273195986286282) q[4];
rz(-2.3905274954577016) q[4];
ry(1.5686645352599902) q[5];
rz(0.9067726605549112) q[5];
ry(-2.256260074717717) q[6];
rz(-1.1352452551418557) q[6];
ry(-2.5182287196677984) q[7];
rz(1.5713743740161634) q[7];
ry(1.5691550535992935) q[8];
rz(2.5026597232375205) q[8];
ry(0.0006050829121629723) q[9];
rz(-2.5071123833631446) q[9];
ry(-1.5637612226564777) q[10];
rz(-0.014053423871101142) q[10];
ry(-0.0022636894958711906) q[11];
rz(0.5401492315244862) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.07260068943733433) q[0];
rz(2.0264548090164913) q[0];
ry(-1.57053048237728) q[1];
rz(-1.569711469845088) q[1];
ry(1.5705237042488003) q[2];
rz(0.9344910672766172) q[2];
ry(-1.5700351249912314) q[3];
rz(1.2363065210122566) q[3];
ry(-3.140560899634799) q[4];
rz(0.9641056896854802) q[4];
ry(-3.1411339006103187) q[5];
rz(1.7477123567166757) q[5];
ry(-0.0004730723868044961) q[6];
rz(0.43037399282017236) q[6];
ry(-0.9833162930245037) q[7];
rz(-0.0006265704267340354) q[7];
ry(3.139211262158529) q[8];
rz(-0.7385136638600489) q[8];
ry(1.5712583292156967) q[9];
rz(-0.4043236270813432) q[9];
ry(-1.5705676524301433) q[10];
rz(0.05419603996103283) q[10];
ry(-1.5340998819494827) q[11];
rz(1.5568379218811774) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5708452812342362) q[0];
rz(1.6688243047309692) q[0];
ry(-1.0234597778615824) q[1];
rz(-0.0012371834397510997) q[1];
ry(0.001548719665640149) q[2];
rz(0.6371206747401365) q[2];
ry(-0.001261811359998753) q[3];
rz(-1.9116390893641952) q[3];
ry(0.001588677292124597) q[4];
rz(1.668879211548732) q[4];
ry(3.1410907058823527) q[5];
rz(0.7338427284269642) q[5];
ry(3.1399824638837543) q[6];
rz(2.2286894032089535) q[6];
ry(1.5706114542175706) q[7];
rz(0.00018192782135351138) q[7];
ry(1.5699327497510371) q[8];
rz(-3.033966821621777) q[8];
ry(-3.131783359350289) q[9];
rz(-0.4038231516948346) q[9];
ry(1.778862086373648) q[10];
rz(-0.46992382662484555) q[10];
ry(1.5705731007584625) q[11];
rz(1.7795671467436476) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.0009346695718237541) q[0];
rz(-0.09860014070914444) q[0];
ry(-1.571406643316382) q[1];
rz(-0.0007236811068988435) q[1];
ry(1.5710077064041386) q[2];
rz(-1.761282263301569) q[2];
ry(0.0017402355919236986) q[3];
rz(2.2456937192616433) q[3];
ry(-3.1403981345625978) q[4];
rz(0.8373297672460916) q[4];
ry(-0.00010171406395597224) q[5];
rz(2.8239013285236183) q[5];
ry(3.1413760039504246) q[6];
rz(1.7990518245473988) q[6];
ry(2.155095260320263) q[7];
rz(-0.11073678711267389) q[7];
ry(-3.137342914487842) q[8];
rz(0.9320557827111806) q[8];
ry(1.5696572910935591) q[9];
rz(0.5764423719479144) q[9];
ry(-0.000638449065558433) q[10];
rz(-1.1265418609946982) q[10];
ry(-3.0185854749599743) q[11];
rz(1.9385717065715886) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.0415332509363613) q[0];
rz(1.1890219912803488) q[0];
ry(-2.117974297882087) q[1];
rz(1.5714849855456716) q[1];
ry(3.1328022646103673) q[2];
rz(2.9487679984457107) q[2];
ry(-1.5709480034873486) q[3];
rz(-0.4684601763330714) q[3];
ry(-1.5747303677746096) q[4];
rz(2.0810756100154304) q[4];
ry(0.056243409915873954) q[5];
rz(1.273018487061072) q[5];
ry(0.43851888605042083) q[6];
rz(-0.15518413754936944) q[6];
ry(1.520124466333138) q[7];
rz(-3.039300992998845) q[7];
ry(-3.1406438317911283) q[8];
rz(-0.7464808564727089) q[8];
ry(-3.13776547921423) q[9];
rz(1.0479153477336638) q[9];
ry(-3.0884161719894276) q[10];
rz(-1.6577152725116715) q[10];
ry(-1.4768995095481547) q[11];
rz(0.0013474422413952253) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.012676405643968103) q[0];
rz(-2.9742926382121806) q[0];
ry(-1.5717453384625504) q[1];
rz(-1.0853822492667904) q[1];
ry(-1.5709354863681597) q[2];
rz(-0.0006989518568585756) q[2];
ry(0.001882578947117603) q[3];
rz(0.0634268403376665) q[3];
ry(-0.0004032121438051206) q[4];
rz(-0.4919744688768519) q[4];
ry(-5.672108680713352e-05) q[5];
rz(0.7216650125682005) q[5];
ry(-0.0007189306925981808) q[6];
rz(-3.0253907671028806) q[6];
ry(-3.140433401403667) q[7];
rz(2.0970731768828017) q[7];
ry(-1.5704117925908916) q[8];
rz(3.138271277301639) q[8];
ry(-0.0022748764038800218) q[9];
rz(1.0993564242265936) q[9];
ry(3.14110708978039) q[10];
rz(-1.7981972907937047) q[10];
ry(-2.479042275416373) q[11];
rz(-0.767910027068083) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.1384283348071698) q[0];
rz(-0.2103423126186257) q[0];
ry(3.1402320858676447) q[1];
rz(1.0575771044511013) q[1];
ry(1.5622829486475498) q[2];
rz(1.5751711384731366) q[2];
ry(3.141536630557336) q[3];
rz(-1.4039123437045748) q[3];
ry(-0.000534840868626363) q[4];
rz(1.556292966361589) q[4];
ry(1.5434516697484533) q[5];
rz(-2.517557536746533) q[5];
ry(1.5707512003814763) q[6];
rz(-3.1377075778729724) q[6];
ry(-3.019766452415535) q[7];
rz(-0.5726954226014183) q[7];
ry(1.5696819971661826) q[8];
rz(1.5748329320370986) q[8];
ry(-1.5709920123963104) q[9];
rz(-2.5699314413328405) q[9];
ry(-3.141465667777908) q[10];
rz(1.4969925629662575) q[10];
ry(-1.6433115417033202) q[11];
rz(2.348496338164994) q[11];