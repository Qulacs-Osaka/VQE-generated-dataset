OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.2350721004949627) q[0];
rz(1.9445374265897861) q[0];
ry(0.7996941591491122) q[1];
rz(2.970426490272006) q[1];
ry(-2.924222629325998) q[2];
rz(3.0508976755740744) q[2];
ry(-1.0094046731478519) q[3];
rz(-0.46894396087965967) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.06625823970994205) q[0];
rz(1.1145471688232875) q[0];
ry(-0.44266749865047506) q[1];
rz(-0.40946794734416897) q[1];
ry(2.8181231084882437) q[2];
rz(1.3260927861067202) q[2];
ry(1.0693400406982503) q[3];
rz(1.745331464695032) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.858585711162378) q[0];
rz(0.942149254972093) q[0];
ry(-2.014675796003381) q[1];
rz(-1.7501475388804508) q[1];
ry(-1.988249234850899) q[2];
rz(-0.17157133885943487) q[2];
ry(-2.0435923465803274) q[3];
rz(1.7655030148034614) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.6440917467995728) q[0];
rz(-0.35756852705019426) q[0];
ry(-2.819846362640799) q[1];
rz(-1.139899593944464) q[1];
ry(-2.4216590393904416) q[2];
rz(3.085254479171155) q[2];
ry(-1.8885645838098772) q[3];
rz(0.5012097539643285) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.457483462661993) q[0];
rz(2.090832675605218) q[0];
ry(0.9315244640091517) q[1];
rz(1.5278510474279514) q[1];
ry(0.8086309771465262) q[2];
rz(1.0208571495045922) q[2];
ry(-1.8586630482628745) q[3];
rz(1.9260007531707242) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.6477296957152845) q[0];
rz(0.7185925880656077) q[0];
ry(-2.1365797399207995) q[1];
rz(2.849144640877811) q[1];
ry(-0.013809111615967316) q[2];
rz(2.823773809570266) q[2];
ry(-2.9129101381094977) q[3];
rz(-0.5898232933911123) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1136969578628948) q[0];
rz(-3.085424410871827) q[0];
ry(-2.032551094305525) q[1];
rz(-2.914262910137989) q[1];
ry(1.7993749789064033) q[2];
rz(-2.67568599691708) q[2];
ry(1.53979311537584) q[3];
rz(-1.9800178211734354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.27849164940153415) q[0];
rz(-1.1794310572538171) q[0];
ry(-0.09846267751122006) q[1];
rz(0.1159453410361381) q[1];
ry(-1.9089873423636758) q[2];
rz(0.6216349266533996) q[2];
ry(-1.892138346178431) q[3];
rz(-2.7080411627294354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.1049045792096197) q[0];
rz(-2.3585707917449907) q[0];
ry(2.302148495523579) q[1];
rz(0.7487000613289815) q[1];
ry(-2.1291973446837984) q[2];
rz(1.0832625155769433) q[2];
ry(-2.8603112083182243) q[3];
rz(0.8500981053403667) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.2846758816037154) q[0];
rz(-1.4416995174900782) q[0];
ry(-1.3615972855872547) q[1];
rz(0.5251964330927965) q[1];
ry(-2.5062144810746663) q[2];
rz(-0.1364397365608656) q[2];
ry(1.2824553518837947) q[3];
rz(-2.212275159361409) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.4167661171084908) q[0];
rz(-2.831808824581918) q[0];
ry(-1.7399097440414986) q[1];
rz(1.529219620324226) q[1];
ry(-2.2585038024398774) q[2];
rz(3.104389128070118) q[2];
ry(-0.8893091115396039) q[3];
rz(-2.7227527586698947) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.6100194838614694) q[0];
rz(1.8367146486011288) q[0];
ry(-1.8089848050666903) q[1];
rz(-0.40438206571391666) q[1];
ry(-1.3948066943062098) q[2];
rz(0.8222009839822555) q[2];
ry(1.221767541249335) q[3];
rz(-0.39950063024384275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.4038485303009116) q[0];
rz(-0.9248564100537235) q[0];
ry(3.101954767205071) q[1];
rz(0.89691135069933) q[1];
ry(0.6482858319291531) q[2];
rz(1.2843651283144126) q[2];
ry(0.484450870727799) q[3];
rz(1.1210339753007434) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7287154356352743) q[0];
rz(-2.9236592799081946) q[0];
ry(-2.8614955824319264) q[1];
rz(-2.8003228681697663) q[1];
ry(1.226915804092575) q[2];
rz(-1.5567946117315) q[2];
ry(0.860479014878935) q[3];
rz(-0.8314129691589831) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3539224567501602) q[0];
rz(1.8834002864968018) q[0];
ry(-1.5817910646584838) q[1];
rz(-3.026355650190598) q[1];
ry(1.1442112898817287) q[2];
rz(2.020875950366409) q[2];
ry(-2.6212260767501103) q[3];
rz(0.957729364800586) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.0683747501383083) q[0];
rz(-0.0012026733202873798) q[0];
ry(-0.263632323850752) q[1];
rz(1.264206856906078) q[1];
ry(1.3114463111662928) q[2];
rz(-0.06786928449344565) q[2];
ry(3.015726027982984) q[3];
rz(2.5836106796575202) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7548511825519024) q[0];
rz(-2.4929900578514754) q[0];
ry(2.125829210296492) q[1];
rz(-0.7595692508861767) q[1];
ry(0.45312102165407664) q[2];
rz(-1.6648302055629536) q[2];
ry(-1.7529688375284525) q[3];
rz(-0.6831303887589559) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8062136471418326) q[0];
rz(-2.604408421933757) q[0];
ry(-0.27435045295931937) q[1];
rz(0.6296951227171043) q[1];
ry(1.803011968117821) q[2];
rz(1.5355424594912215) q[2];
ry(2.1573225443788964) q[3];
rz(0.0687697998405792) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.12932331424528076) q[0];
rz(1.3823765440830895) q[0];
ry(-0.0373994988332802) q[1];
rz(1.0954223616844843) q[1];
ry(-0.3092227363878095) q[2];
rz(-2.42534906499496) q[2];
ry(-1.0392733191750354) q[3];
rz(-2.232030071409981) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.4543669356641904) q[0];
rz(-2.502330406119307) q[0];
ry(-1.8755875408041247) q[1];
rz(-0.2374670323952186) q[1];
ry(2.1344649378554226) q[2];
rz(0.10940315545731938) q[2];
ry(0.6512906778253409) q[3];
rz(-1.8524430705782375) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7425154338166813) q[0];
rz(2.457926638817249) q[0];
ry(0.3624555581876825) q[1];
rz(-1.8239585558008151) q[1];
ry(1.5979594483224178) q[2];
rz(-3.12732906166492) q[2];
ry(1.1336420400346465) q[3];
rz(1.1350247914883727) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1436467167343736) q[0];
rz(0.7677945832739654) q[0];
ry(1.290910970324389) q[1];
rz(1.467757828637942) q[1];
ry(1.4081922637055) q[2];
rz(-1.5951228995386921) q[2];
ry(-2.6317166532578486) q[3];
rz(-1.4553313226648297) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.5204127721626004) q[0];
rz(-0.28273655416838056) q[0];
ry(0.09438432263646311) q[1];
rz(1.8716295912138938) q[1];
ry(-1.1344891935944634) q[2];
rz(-2.128754896432838) q[2];
ry(-2.1272625383833748) q[3];
rz(-2.231562778929589) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.74335838217239) q[0];
rz(-1.6837455010007023) q[0];
ry(-0.23716775531916037) q[1];
rz(-2.768738779175659) q[1];
ry(1.1525839394135016) q[2];
rz(0.40459448354187794) q[2];
ry(0.8221920275125827) q[3];
rz(-2.008244286609699) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.0458832452829574) q[0];
rz(-0.08869538418002187) q[0];
ry(-1.7659158934199723) q[1];
rz(0.8130979489836969) q[1];
ry(-2.386690201122287) q[2];
rz(2.661009087309287) q[2];
ry(-0.019882768960416186) q[3];
rz(-1.9306932310805787) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.2844109150425747) q[0];
rz(-1.2654681576163673) q[0];
ry(-1.6657404011405703) q[1];
rz(-1.5938514458037005) q[1];
ry(1.8638102326451103) q[2];
rz(-3.042469687441401) q[2];
ry(1.9588555756814374) q[3];
rz(0.28634466684468785) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8789157176140636) q[0];
rz(1.4948298820999275) q[0];
ry(1.5490659777930982) q[1];
rz(-2.9927060621748764) q[1];
ry(0.32965793063522497) q[2];
rz(-2.7668203896507233) q[2];
ry(3.0603366360706383) q[3];
rz(-0.756626790605293) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7540808736653313) q[0];
rz(2.780663399047946) q[0];
ry(0.7017360497638876) q[1];
rz(2.30812429462638) q[1];
ry(1.2836334051546423) q[2];
rz(-1.6817250371421903) q[2];
ry(-1.31052811577728) q[3];
rz(-0.5048751624745176) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.481891966607231) q[0];
rz(0.4413095388369923) q[0];
ry(-2.3374824128041682) q[1];
rz(-2.6908043558542043) q[1];
ry(-0.6811829898117621) q[2];
rz(-0.7684879534170042) q[2];
ry(-2.575039073977648) q[3];
rz(1.9298759964604462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.8549482067872289) q[0];
rz(-0.24888652201838646) q[0];
ry(-0.09167258203719297) q[1];
rz(1.6050792889805596) q[1];
ry(1.1325935674214753) q[2];
rz(-0.629680036641481) q[2];
ry(-1.156380941167935) q[3];
rz(-0.007734927014172353) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.17206893826021655) q[0];
rz(-1.111211258823146) q[0];
ry(1.5167542006932027) q[1];
rz(-3.0790549606088793) q[1];
ry(0.9358323460966913) q[2];
rz(-1.3743208011307007) q[2];
ry(-2.216213576410995) q[3];
rz(-0.3694378471628044) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.363123964328806) q[0];
rz(2.922894356650476) q[0];
ry(-1.571222955021045) q[1];
rz(1.4154597724564852) q[1];
ry(-0.6231628934752047) q[2];
rz(-0.671299241122905) q[2];
ry(-1.2757725989543216) q[3];
rz(-2.55510821204371) q[3];