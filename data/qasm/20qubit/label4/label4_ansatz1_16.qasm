OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.5729591593926591) q[0];
rz(-1.3474119061491612) q[0];
ry(1.5233848037464472) q[1];
rz(-2.2720255582112943) q[1];
ry(2.445916482256052) q[2];
rz(-0.567008907233725) q[2];
ry(0.007626133620465408) q[3];
rz(-3.1017083889788344) q[3];
ry(1.539382493571154) q[4];
rz(1.658449540859638) q[4];
ry(-1.5921243741596707) q[5];
rz(2.8327047261276888) q[5];
ry(0.44195202587712706) q[6];
rz(-0.3352719288955039) q[6];
ry(-0.22354091342557061) q[7];
rz(2.8556283225314885) q[7];
ry(1.5599192666338242) q[8];
rz(-1.7537185051954935) q[8];
ry(-1.1266570497388466) q[9];
rz(3.047594273823697) q[9];
ry(-0.16447918518945492) q[10];
rz(-0.1251644672697001) q[10];
ry(3.133730263665162) q[11];
rz(-0.5996509331109946) q[11];
ry(0.020493361970550517) q[12];
rz(3.028813216925588) q[12];
ry(-0.3598144825530225) q[13];
rz(0.4984994570824437) q[13];
ry(0.5549914364102574) q[14];
rz(-1.508197249381369) q[14];
ry(1.5888777866945754) q[15];
rz(-1.364214657848302) q[15];
ry(0.010154004764331148) q[16];
rz(1.9785519142246466) q[16];
ry(1.9197682013284296) q[17];
rz(-0.6659312431606779) q[17];
ry(-1.581258675639676) q[18];
rz(2.0925344520008666) q[18];
ry(-0.8210260660349453) q[19];
rz(2.3211388825455908) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.028753669442267) q[0];
rz(-1.4880030418658505) q[0];
ry(1.5881838481290786) q[1];
rz(1.7424515665214382) q[1];
ry(-0.344717054992679) q[2];
rz(0.1920459085575457) q[2];
ry(3.1296134244022826) q[3];
rz(-3.059268152630796) q[3];
ry(1.8715264527773332) q[4];
rz(2.8009223380020014) q[4];
ry(1.576808402112218) q[5];
rz(-1.4901115681712982) q[5];
ry(-0.1314432520086246) q[6];
rz(1.089990634660664) q[6];
ry(1.5583711755340024) q[7];
rz(0.03348340993026843) q[7];
ry(3.056017852017579) q[8];
rz(-1.747593638035858) q[8];
ry(3.1243523316712913) q[9];
rz(3.1202043369552594) q[9];
ry(3.1402109140637244) q[10];
rz(0.44074807085153006) q[10];
ry(0.022798238660221234) q[11];
rz(-0.5853971626789836) q[11];
ry(-0.8619065383811044) q[12];
rz(1.0571565158083773) q[12];
ry(0.3843667999225229) q[13];
rz(0.9527768447874784) q[13];
ry(-0.012850574772330342) q[14];
rz(-1.5061592574763942) q[14];
ry(0.054938735571663386) q[15];
rz(-0.04563916867280108) q[15];
ry(2.2952572706513625) q[16];
rz(0.07998748579079977) q[16];
ry(0.8237554155944248) q[17];
rz(-2.0334120888816947) q[17];
ry(-1.491878265054134) q[18];
rz(-1.6928507384405433) q[18];
ry(3.0114477036777436) q[19];
rz(1.7185475078006283) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.6669005825090657) q[0];
rz(-0.8231608914577473) q[0];
ry(0.7493260853516973) q[1];
rz(0.03902285086116741) q[1];
ry(1.2992903373958793) q[2];
rz(0.31976037201716934) q[2];
ry(0.021345743487423834) q[3];
rz(0.6053945894811035) q[3];
ry(-2.978969185004614) q[4];
rz(-0.6502538387269476) q[4];
ry(1.540978996218385) q[5];
rz(2.579731306270155) q[5];
ry(1.565170268219414) q[6];
rz(0.046264433914342686) q[6];
ry(1.5067388548532383) q[7];
rz(0.004428555047580218) q[7];
ry(-1.275852388501658) q[8];
rz(3.105200781928587) q[8];
ry(-1.9905713847432693) q[9];
rz(-0.36352449594571934) q[9];
ry(-0.09538178460602875) q[10];
rz(-1.559429237150904) q[10];
ry(3.1347650576889676) q[11];
rz(1.8525535947301877) q[11];
ry(1.7806368472243141) q[12];
rz(1.4305342712813258) q[12];
ry(-0.14486891213507389) q[13];
rz(0.3664805022515684) q[13];
ry(-2.7871354241107458) q[14];
rz(3.101332194293834) q[14];
ry(2.9665770037157024) q[15];
rz(-2.8507807445426643) q[15];
ry(0.11173531735203125) q[16];
rz(2.3408176237823275) q[16];
ry(-1.8658375133801153) q[17];
rz(-1.0443121807011877) q[17];
ry(-3.1227944079999466) q[18];
rz(1.0578017077560085) q[18];
ry(2.22374027102281) q[19];
rz(2.8251663395866067) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.0271686807969478) q[0];
rz(3.0138920385335175) q[0];
ry(-2.5861560724463444) q[1];
rz(2.8867118220624914) q[1];
ry(-1.9032058885308052) q[2];
rz(0.2860600517294838) q[2];
ry(-2.986033858734731) q[3];
rz(-0.4409796626795679) q[3];
ry(-0.28675307497570235) q[4];
rz(2.7937700673916055) q[4];
ry(0.028268679190757595) q[5];
rz(-0.5073812732324869) q[5];
ry(-1.5724544150806004) q[6];
rz(-1.6742449432748276) q[6];
ry(1.5692842881471978) q[7];
rz(2.971964114469694) q[7];
ry(-1.5797328407020714) q[8];
rz(-0.8081815571597079) q[8];
ry(-1.5364347158516853) q[9];
rz(-0.3081052006417204) q[9];
ry(1.4892193167416474) q[10];
rz(-3.0829582192704623) q[10];
ry(-1.37894441745218) q[11];
rz(2.9102240078825528) q[11];
ry(-2.4480427190751963) q[12];
rz(-2.2824209060301683) q[12];
ry(-0.6901778546373852) q[13];
rz(-1.1681496157679399) q[13];
ry(-1.8575353271334083) q[14];
rz(-0.23703837910688288) q[14];
ry(-3.035018987274132) q[15];
rz(-1.8063708932060507) q[15];
ry(-0.7837491312792118) q[16];
rz(0.7246977258865634) q[16];
ry(-0.3996398979739342) q[17];
rz(-2.640519131011798) q[17];
ry(0.7744576077812759) q[18];
rz(-1.907652919355546) q[18];
ry(-1.9684741730434805) q[19];
rz(-2.7403409467074735) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.324756553328322) q[0];
rz(-0.8581770914797598) q[0];
ry(-1.8154042590254669) q[1];
rz(2.023022484235129) q[1];
ry(1.0579979577219194) q[2];
rz(0.3925763749655289) q[2];
ry(2.9572409264323416) q[3];
rz(-0.9512239197795969) q[3];
ry(-2.9612728434742106) q[4];
rz(-2.240115122928234) q[4];
ry(3.0976696724787853) q[5];
rz(0.678990225391649) q[5];
ry(-1.82720810770932) q[6];
rz(-0.222115423396132) q[6];
ry(-2.7040411165325366) q[7];
rz(-0.5551842461483919) q[7];
ry(1.5433459153604634) q[8];
rz(1.6563567761524745) q[8];
ry(0.01616635731739091) q[9];
rz(0.30581405506162096) q[9];
ry(0.13417997533098047) q[10];
rz(-1.6486177997642883) q[10];
ry(-1.675590818295273) q[11];
rz(0.014280122239244797) q[11];
ry(-0.17569284786242584) q[12];
rz(-3.0704525673394265) q[12];
ry(3.120074769660309) q[13];
rz(2.5423010394505705) q[13];
ry(0.2192091935392817) q[14];
rz(2.0914847605843825) q[14];
ry(3.1377458701492227) q[15];
rz(-1.3025655708578148) q[15];
ry(-0.20805162578014508) q[16];
rz(0.8247341754064209) q[16];
ry(-2.953150796333804) q[17];
rz(-0.939993850435391) q[17];
ry(-1.1898598521676778) q[18];
rz(2.739108672615472) q[18];
ry(-1.0963681034322923) q[19];
rz(2.7271617616581545) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.182386993807266) q[0];
rz(0.3810945151787481) q[0];
ry(-2.0976305928163246) q[1];
rz(-0.6277633475830254) q[1];
ry(-0.05929277484227212) q[2];
rz(-0.24250983770950726) q[2];
ry(2.9130165036026097) q[3];
rz(2.193121662659596) q[3];
ry(1.9231997099178395) q[4];
rz(-0.2604026711251643) q[4];
ry(-3.0671914230959767) q[5];
rz(-0.812899012367664) q[5];
ry(0.009071703819076404) q[6];
rz(0.27080978168002456) q[6];
ry(3.1415846421560287) q[7];
rz(-0.3104164866311938) q[7];
ry(-0.0005956025129423281) q[8];
rz(-1.6577773303837977) q[8];
ry(-1.5671639028360298) q[9];
rz(-1.5622184016636051) q[9];
ry(0.48349971509366263) q[10];
rz(-3.0861193143007872) q[10];
ry(-2.2825814641133233) q[11];
rz(-2.2686629685755912) q[11];
ry(-3.0803894753525043) q[12];
rz(0.06395858653940854) q[12];
ry(-0.7526925827032563) q[13];
rz(-2.967086822595331) q[13];
ry(-0.7844811709019411) q[14];
rz(-0.37944682879895986) q[14];
ry(-0.7570359144038067) q[15];
rz(2.488594226243366) q[15];
ry(0.6690001080375736) q[16];
rz(-1.2185292958286107) q[16];
ry(-2.134472837613692) q[17];
rz(-1.0170389319294628) q[17];
ry(2.3604630139615006) q[18];
rz(-0.5466588121481383) q[18];
ry(-2.6305433214182106) q[19];
rz(2.5405868007474486) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.9305028200379344) q[0];
rz(1.5280065503500184) q[0];
ry(2.4502466659912523) q[1];
rz(1.3855509527080294) q[1];
ry(-0.15938556143484692) q[2];
rz(-2.9265853688971726) q[2];
ry(-0.6568142020002803) q[3];
rz(1.7652326971099812) q[3];
ry(-3.121925421552884) q[4];
rz(0.024928495683086638) q[4];
ry(-0.1209963523575898) q[5];
rz(2.1462311751120104) q[5];
ry(1.8469418747423492) q[6];
rz(0.38519143502876396) q[6];
ry(1.1513412827918357) q[7];
rz(2.026242244024372) q[7];
ry(1.5317675281395617) q[8];
rz(-1.5719430607865728) q[8];
ry(-1.5896660075815172) q[9];
rz(-2.6294485436428214) q[9];
ry(-2.9654598220905237) q[10];
rz(-0.014104342708201318) q[10];
ry(3.1313265256034515) q[11];
rz(-1.1015904094917461) q[11];
ry(-1.6894630566301876) q[12];
rz(-3.0866073390025988) q[12];
ry(2.476281038803737) q[13];
rz(1.020182588337008) q[13];
ry(-0.03952254894739049) q[14];
rz(2.0174455100831614) q[14];
ry(-1.5349110061497768) q[15];
rz(0.05498201164765604) q[15];
ry(2.9696473679135016) q[16];
rz(-0.12732683077090456) q[16];
ry(2.8092348952563073) q[17];
rz(0.936267619894239) q[17];
ry(-1.6126559135117908) q[18];
rz(0.23400671520746205) q[18];
ry(1.6431074547885949) q[19];
rz(2.613088453783465) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.052058522718766) q[0];
rz(2.265680361801979) q[0];
ry(-0.1735922452819715) q[1];
rz(-0.1819122854024222) q[1];
ry(-3.1363094440469106) q[2];
rz(-1.0841436963862874) q[2];
ry(0.06227786577489525) q[3];
rz(-2.393795701064303) q[3];
ry(-0.05689146114862376) q[4];
rz(2.9056990549968202) q[4];
ry(-1.4329493881047142) q[5];
rz(-2.9526293686758596) q[5];
ry(0.117159713140496) q[6];
rz(-1.0896685718356083) q[6];
ry(-1.680527423902594) q[7];
rz(-1.2849546833347838) q[7];
ry(1.5783595332624294) q[8];
rz(0.15843265531231143) q[8];
ry(-0.0290294035242956) q[9];
rz(-1.7732936273290516) q[9];
ry(1.6545045305028414) q[10];
rz(0.10156792574916824) q[10];
ry(0.0015154255105718601) q[11];
rz(-2.481889487967096) q[11];
ry(3.140152624316458) q[12];
rz(0.9893680387657517) q[12];
ry(2.335796244454983) q[13];
rz(0.9160933468438122) q[13];
ry(0.0011160634509961298) q[14];
rz(-2.059770821446892) q[14];
ry(1.9277586558147188) q[15];
rz(2.2699125539519867) q[15];
ry(-0.7339938404904832) q[16];
rz(2.9055857006296018) q[16];
ry(-0.8406340095200172) q[17];
rz(-0.3336314472887218) q[17];
ry(1.8863083866916446) q[18];
rz(-1.3896961334038966) q[18];
ry(-1.805322164319275) q[19];
rz(1.4662954938140937) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.6537134598746965) q[0];
rz(-1.2641922242498473) q[0];
ry(-0.9118232758037772) q[1];
rz(1.9569724812982212) q[1];
ry(2.9987300021927195) q[2];
rz(2.2540563871174117) q[2];
ry(-1.2418087273640672) q[3];
rz(-2.430744442055441) q[3];
ry(0.07284070838574551) q[4];
rz(-0.919993724582027) q[4];
ry(-2.9724780478625803) q[5];
rz(2.447149738503764) q[5];
ry(-3.1396752646864803) q[6];
rz(2.3168881026470207) q[6];
ry(0.5657682785454581) q[7];
rz(1.8109912926835978) q[7];
ry(-2.0121897290464674) q[8];
rz(-3.0755049770792495) q[8];
ry(0.45879374913822896) q[9];
rz(0.5731703718382422) q[9];
ry(1.459057555718026) q[10];
rz(0.08700553010555012) q[10];
ry(-3.127146221294361) q[11];
rz(0.3173254594620625) q[11];
ry(-3.126639296099349) q[12];
rz(0.9555965757740458) q[12];
ry(2.4705777451634323) q[13];
rz(0.6931505355368746) q[13];
ry(2.870563409266811) q[14];
rz(3.1393274574567798) q[14];
ry(-3.070864193071182) q[15];
rz(3.0367765267265274) q[15];
ry(0.03532430678464493) q[16];
rz(-2.903486114864915) q[16];
ry(-0.03894034848076977) q[17];
rz(-3.1366939517368673) q[17];
ry(-1.885463015555782) q[18];
rz(-2.926767452775024) q[18];
ry(1.841171972458989) q[19];
rz(-0.2559299483603143) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.162341644677432) q[0];
rz(-2.389811185509713) q[0];
ry(2.937746008863112) q[1];
rz(-1.9786833020539385) q[1];
ry(0.1810192959645758) q[2];
rz(-0.2790136372351253) q[2];
ry(2.0542231810337137) q[3];
rz(0.6000244464664561) q[3];
ry(0.03051040829762737) q[4];
rz(-1.3578900704457617) q[4];
ry(0.15071004421884027) q[5];
rz(-0.736775759701529) q[5];
ry(1.9181705063939738) q[6];
rz(-3.1399253543187085) q[6];
ry(3.134397729743848) q[7];
rz(1.9355271481363638) q[7];
ry(-3.1232958404328346) q[8];
rz(-3.0753610320832823) q[8];
ry(3.134925884044368) q[9];
rz(2.547657200362827) q[9];
ry(-0.029592428879452903) q[10];
rz(-2.2041013755778445) q[10];
ry(0.005147921966991037) q[11];
rz(1.516872996388269) q[11];
ry(3.030954732072902) q[12];
rz(-0.9778033971764774) q[12];
ry(-1.3258409546595848) q[13];
rz(-1.3736968398428921) q[13];
ry(-1.4566698679949759) q[14];
rz(-0.0021082185029707035) q[14];
ry(-0.013498068130452633) q[15];
rz(-0.8079157626571201) q[15];
ry(2.396484054818746) q[16];
rz(3.0837405943982565) q[16];
ry(-1.8031715326557363) q[17];
rz(-1.198777800849128) q[17];
ry(2.437834566589439) q[18];
rz(-2.080735655950872) q[18];
ry(0.5505999255859589) q[19];
rz(-0.6861485254006389) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.3699920552760727) q[0];
rz(-1.3758665553876543) q[0];
ry(2.4635563488469594) q[1];
rz(0.7210850592648078) q[1];
ry(1.17675785695986) q[2];
rz(-3.1202186174506172) q[2];
ry(2.862096604474705) q[3];
rz(0.5892798257901378) q[3];
ry(-0.0010333751994976534) q[4];
rz(-0.9082478652909243) q[4];
ry(-1.595710147020192) q[5];
rz(-0.11244775891493379) q[5];
ry(-0.6887483845778591) q[6];
rz(0.0027810442040893607) q[6];
ry(-2.7648875561658595) q[7];
rz(0.21369736016655508) q[7];
ry(2.0138269654883842) q[8];
rz(0.19730528038169065) q[8];
ry(-1.5753169873187476) q[9];
rz(2.712784192351531) q[9];
ry(-0.11759435693117645) q[10];
rz(2.1020585416379127) q[10];
ry(1.607211588399588) q[11];
rz(-0.8721882675738788) q[11];
ry(-3.1345279271045268) q[12];
rz(2.160182542992133) q[12];
ry(1.5086588626023119) q[13];
rz(-0.002209804839490081) q[13];
ry(-1.7204862061637836) q[14];
rz(0.0003736148047611868) q[14];
ry(1.6732338436167824) q[15];
rz(2.931138584691098) q[15];
ry(2.741552498538602) q[16];
rz(2.495607896128207) q[16];
ry(-1.846852134151887) q[17];
rz(2.9701590187135873) q[17];
ry(-1.2304924069435368) q[18];
rz(-1.2681103405774925) q[18];
ry(-1.1334592714991576) q[19];
rz(-2.8881182828020804) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.4337997981376285) q[0];
rz(1.2035497331006961) q[0];
ry(-0.7673922580561817) q[1];
rz(-2.9892107046983147) q[1];
ry(1.8449728801440735) q[2];
rz(0.9306791834574978) q[2];
ry(-2.2480313791447157) q[3];
rz(3.090459287793287) q[3];
ry(-2.1169682281966065) q[4];
rz(-0.0017369469147984953) q[4];
ry(0.08793576674648218) q[5];
rz(-0.042988832442528616) q[5];
ry(1.545910910597677) q[6];
rz(-3.141180409425112) q[6];
ry(2.702387787964376) q[7];
rz(2.6507533903564635) q[7];
ry(3.0074547018620366) q[8];
rz(0.036000914753241804) q[8];
ry(1.6296955888201814) q[9];
rz(-1.821952088543145) q[9];
ry(-1.8057091943627552) q[10];
rz(-1.7866216782185997) q[10];
ry(3.1006143174430822) q[11];
rz(-0.8900863508064599) q[11];
ry(2.9109477646024837) q[12];
rz(3.1054493750773045) q[12];
ry(-1.5369403141769524) q[13];
rz(0.17808695488650716) q[13];
ry(-1.697130381770348) q[14];
rz(-1.7263302805541254) q[14];
ry(-0.0009618572905096283) q[15];
rz(-2.9973246274948617) q[15];
ry(-0.001729058957681495) q[16];
rz(0.6420140685236506) q[16];
ry(-1.2773143059495045) q[17];
rz(-0.737115965647912) q[17];
ry(1.2546387172634696) q[18];
rz(-0.09684608007842765) q[18];
ry(1.293654868469372) q[19];
rz(-0.09913575994008195) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.060371329716938) q[0];
rz(0.06861646271659971) q[0];
ry(-3.0654739090301497) q[1];
rz(-2.4845505011557694) q[1];
ry(0.026775777134577374) q[2];
rz(-0.9558219065382554) q[2];
ry(-1.0958976559998668) q[3];
rz(-3.1361212860394487) q[3];
ry(-2.1077716463458644) q[4];
rz(-0.00019822326293983394) q[4];
ry(-1.7392210983837648) q[5];
rz(2.4028758771946395) q[5];
ry(2.2167215111456855) q[6];
rz(-3.116010391104104) q[6];
ry(-3.136393237651615) q[7];
rz(2.502806297173887) q[7];
ry(-2.805673455702145) q[8];
rz(-1.1622331506187058) q[8];
ry(-3.139506336723946) q[9];
rz(1.3923510309402385) q[9];
ry(3.141048678466399) q[10];
rz(-1.7868828078726642) q[10];
ry(1.3816528142987954) q[11];
rz(-0.6728829688872986) q[11];
ry(-3.062565628790519) q[12];
rz(3.106872583303061) q[12];
ry(-1.4631291448399344) q[13];
rz(-0.5223811105267053) q[13];
ry(-0.7826117304674528) q[14];
rz(1.146791798273977) q[14];
ry(-1.861358941796808) q[15];
rz(-1.49926763509581) q[15];
ry(-1.6109459926121086) q[16];
rz(0.0021746520143111066) q[16];
ry(3.042553535410304) q[17];
rz(-2.133631634018472) q[17];
ry(-2.035146716419415) q[18];
rz(-2.1329072650177334) q[18];
ry(3.0787966685174757) q[19];
rz(-1.982066868164546) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.932490817100436) q[0];
rz(0.7798012778428101) q[0];
ry(-2.152326180408309) q[1];
rz(0.017581301623023144) q[1];
ry(2.942496019842482) q[2];
rz(-0.025817121434209334) q[2];
ry(2.5686827871684335) q[3];
rz(-0.3149385205790143) q[3];
ry(-0.4903092545206732) q[4];
rz(2.826678682742019) q[4];
ry(-3.137810069159687) q[5];
rz(-0.7437325898858695) q[5];
ry(-0.15965156361404897) q[6];
rz(-2.6415261436391497) q[6];
ry(-1.3621163968618724) q[7];
rz(-1.9771033251134627) q[7];
ry(-0.00498048641816378) q[8];
rz(1.1566654968703693) q[8];
ry(0.04324561498321433) q[9];
rz(3.068400640156459) q[9];
ry(-1.5792060894415816) q[10];
rz(-0.002470521168235429) q[10];
ry(0.002143576632054539) q[11];
rz(0.7140118150869799) q[11];
ry(-2.656056866067359) q[12];
rz(-3.1400376952406104) q[12];
ry(5.6735170635313636e-05) q[13];
rz(-2.6114647399024697) q[13];
ry(-3.135732337604032) q[14];
rz(-0.7913409985601507) q[14];
ry(-3.14008198513871) q[15];
rz(1.707738079144982) q[15];
ry(2.6674200605690146) q[16];
rz(0.942019683938775) q[16];
ry(-3.1066945132554) q[17];
rz(-1.4614366669119612) q[17];
ry(3.133494218580455) q[18];
rz(0.9296325680845257) q[18];
ry(1.3932865909806393) q[19];
rz(1.42917638232392) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.03674930816524391) q[0];
rz(0.054775916431480914) q[0];
ry(-0.0844918341145577) q[1];
rz(-0.4374858122660594) q[1];
ry(0.7065066115179938) q[2];
rz(-2.9685209812237816) q[2];
ry(-2.1313472624420875) q[3];
rz(-0.8890477258327121) q[3];
ry(-3.1388562094999153) q[4];
rz(-1.9575528244358478) q[4];
ry(0.6835402313983645) q[5];
rz(-3.13536917700235) q[5];
ry(-0.0050086429601243765) q[6];
rz(2.6176917916357327) q[6];
ry(-3.0698453053289443) q[7];
rz(-0.9389475320302808) q[7];
ry(1.4174526147086117) q[8];
rz(-3.137559199440788) q[8];
ry(-1.6840810824322614) q[9];
rz(3.1620447717959385e-05) q[9];
ry(-0.12197906574133259) q[10];
rz(0.0014493299749457939) q[10];
ry(-2.289828760099494) q[11];
rz(-3.075317062639952) q[11];
ry(-1.6701606478083286) q[12];
rz(-3.126361916253933) q[12];
ry(-3.042658570589482) q[13];
rz(-1.6120587408509301) q[13];
ry(1.421030964115932) q[14];
rz(-2.3584862247310303) q[14];
ry(0.030574903657901855) q[15];
rz(1.2475765085874146) q[15];
ry(-0.007669146886502403) q[16];
rz(-0.9527285436148227) q[16];
ry(1.5023734213425748) q[17];
rz(2.0703208431425777) q[17];
ry(2.883038481071101) q[18];
rz(1.454209835232552) q[18];
ry(1.799970319136226) q[19];
rz(0.2245271786683496) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.031261261861757) q[0];
rz(-1.4328748361417794) q[0];
ry(-0.31285375143849264) q[1];
rz(-0.517270741213621) q[1];
ry(-3.1353157395948705) q[2];
rz(-2.9950726261440432) q[2];
ry(3.1345255465524167) q[3];
rz(-1.4994736526217998) q[3];
ry(-3.1404027469482116) q[4];
rz(-1.6393747300050425) q[4];
ry(-0.39640127402528724) q[5];
rz(2.9407521542504074) q[5];
ry(1.2843022862378497) q[6];
rz(0.003445129206900921) q[6];
ry(-3.1272266191372617) q[7];
rz(1.0871624077903235) q[7];
ry(2.755494344820324) q[8];
rz(0.003206009450211767) q[8];
ry(-2.751055971151794) q[9];
rz(-0.36490968963995923) q[9];
ry(-1.6403797962424698) q[10];
rz(-0.0033997446697089377) q[10];
ry(2.7766024979318984) q[11];
rz(3.1059632848261955) q[11];
ry(3.092420189237179) q[12];
rz(0.2033246653627861) q[12];
ry(-2.5758349813735806) q[13];
rz(-0.751549022755673) q[13];
ry(-1.81704591103753) q[14];
rz(-0.12171679121142673) q[14];
ry(-3.1375839170701494) q[15];
rz(1.1410193934904802) q[15];
ry(-3.096264824413982) q[16];
rz(-0.5245215476945667) q[16];
ry(-3.1400722405357877) q[17];
rz(0.042051433297716566) q[17];
ry(0.12811472922080203) q[18];
rz(-2.557807070331273) q[18];
ry(2.4624925148522445) q[19];
rz(3.008032848666646) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.5136987343493304) q[0];
rz(0.00590083443850986) q[0];
ry(0.030575872994348607) q[1];
rz(1.187693347818434) q[1];
ry(2.2481531895941442) q[2];
rz(-0.01393708784505332) q[2];
ry(2.7728038597107116) q[3];
rz(2.3122584189608713) q[3];
ry(2.1970150202034513) q[4];
rz(0.0018865421241471838) q[4];
ry(-3.064296326816185) q[5];
rz(2.93001494812797) q[5];
ry(1.038347166305936) q[6];
rz(-3.1396136210251155) q[6];
ry(2.8300358686996727) q[7];
rz(2.878730020686661) q[7];
ry(1.4826698351265304) q[8];
rz(2.4839384335186665) q[8];
ry(-3.1250987471234204) q[9];
rz(2.525115583274751) q[9];
ry(1.5264679159178944) q[10];
rz(-3.1411920128650164) q[10];
ry(1.798344473945788) q[11];
rz(1.6687587405255884) q[11];
ry(-3.136620101759167) q[12];
rz(-3.055168191161047) q[12];
ry(1.6570174194082439) q[13];
rz(-3.031534546571233) q[13];
ry(3.0708382186963052) q[14];
rz(2.2608075854345326) q[14];
ry(1.5583596465617617) q[15];
rz(-0.0012142712459405974) q[15];
ry(0.009564973291767826) q[16];
rz(0.5134084958459635) q[16];
ry(3.1038164608376047) q[17];
rz(1.1633181101224306) q[17];
ry(-1.5494751762010728) q[18];
rz(-1.4710402741140283) q[18];
ry(1.7114884192452147) q[19];
rz(2.3653035197541787) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.3217329367543682) q[0];
rz(-0.7714463634567066) q[0];
ry(-3.1167985620803247) q[1];
rz(3.0669012132189386) q[1];
ry(-1.3241752160551539) q[2];
rz(-2.3615076752720516) q[2];
ry(1.5763589842727495) q[3];
rz(-1.7039117323158912) q[3];
ry(0.9067455524114082) q[4];
rz(-3.1034971167321888) q[4];
ry(2.711971462457912) q[5];
rz(3.127478037501598) q[5];
ry(-1.8062251237006182) q[6];
rz(0.0019386514759587527) q[6];
ry(-3.1125707432887775) q[7];
rz(2.014860609958352) q[7];
ry(-2.696785966553627) q[8];
rz(2.2784096622187056) q[8];
ry(-3.0713118935710337) q[9];
rz(1.5144756728713875) q[9];
ry(0.41252927831021147) q[10];
rz(-3.117020796768494) q[10];
ry(0.0463245632212278) q[11];
rz(-2.851160794917898) q[11];
ry(3.1292041794747494) q[12];
rz(-0.10255010805814424) q[12];
ry(-2.4456387798087635) q[13];
rz(-1.426195964167884) q[13];
ry(-0.0021373781469238295) q[14];
rz(0.7548003280716049) q[14];
ry(0.32011437006791976) q[15];
rz(-0.6858089129693898) q[15];
ry(2.2369294759389664) q[16];
rz(-0.0021049313030272925) q[16];
ry(3.130505806718217) q[17];
rz(-2.4388233157098567) q[17];
ry(-2.8797943925338285) q[18];
rz(-3.0811581859370576) q[18];
ry(-0.0660927549444894) q[19];
rz(-0.779269434486654) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.0077308986138496) q[0];
rz(-2.3872446040592945) q[0];
ry(1.5706382318862844) q[1];
rz(1.5629317981579147) q[1];
ry(-0.0015673155400497407) q[2];
rz(-2.355369039433052) q[2];
ry(0.03303583197834091) q[3];
rz(0.13019711695643288) q[3];
ry(0.010884334344350322) q[4];
rz(1.5318804771259371) q[4];
ry(-1.7355107667043006) q[5];
rz(1.587525382867435) q[5];
ry(-0.575348286606127) q[6];
rz(-1.5701499762184206) q[6];
ry(-0.005371726545168437) q[7];
rz(2.4250783710116233) q[7];
ry(-0.003156802752470797) q[8];
rz(1.824999288803185) q[8];
ry(-3.137841576790374) q[9];
rz(0.19888509711138153) q[9];
ry(-0.052892487805746846) q[10];
rz(-1.5958800956753887) q[10];
ry(0.1036461645818872) q[11];
rz(-0.4012563920631829) q[11];
ry(-1.6025222572001958) q[12];
rz(1.5716725435144134) q[12];
ry(2.143398905489711) q[13];
rz(0.12013606220322474) q[13];
ry(-1.5712139208410791) q[14];
rz(-1.5663215208746715) q[14];
ry(-3.138228989965758) q[15];
rz(0.8806589449411232) q[15];
ry(-1.5813808082245213) q[16];
rz(-1.569735032402891) q[16];
ry(3.134297793925864) q[17];
rz(2.2251007969748064) q[17];
ry(0.9939507447796777) q[18];
rz(0.039838649612027766) q[18];
ry(2.7551627675727453) q[19];
rz(-2.2225593931221486) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.5951168976516044) q[0];
rz(-0.025495834068920864) q[0];
ry(-1.4824456583536945) q[1];
rz(-0.9843677176007474) q[1];
ry(-1.5703595390146523) q[2];
rz(2.758189166974633) q[2];
ry(-1.5835323053457755) q[3];
rz(0.5531848613089784) q[3];
ry(1.5703183711595654) q[4];
rz(0.613249441961001) q[4];
ry(-1.5630285186584512) q[5];
rz(0.9230179573503117) q[5];
ry(1.571783022553168) q[6];
rz(-2.942050312648035) q[6];
ry(-1.5629751513004884) q[7];
rz(0.5120862725418718) q[7];
ry(1.305572560218653) q[8];
rz(0.8506797380817248) q[8];
ry(-1.5271881594156085) q[9];
rz(-1.0981585746065858) q[9];
ry(1.5699838348465598) q[10];
rz(-1.7153121448066209) q[10];
ry(1.5722170039410757) q[11];
rz(2.1106620599978956) q[11];
ry(-1.5715955188879658) q[12];
rz(1.6367936994024312) q[12];
ry(-1.5677834856406032) q[13];
rz(-2.600444367824332) q[13];
ry(-1.5667385297337066) q[14];
rz(1.528845190579342) q[14];
ry(-1.5703856477318103) q[15];
rz(0.5506972334963214) q[15];
ry(-1.5716135319683522) q[16];
rz(-2.261149786742344) q[16];
ry(1.5246856165999607) q[17];
rz(2.1211420658683244) q[17];
ry(-1.3297420855479734) q[18];
rz(-0.026568574443951217) q[18];
ry(0.006985279962619867) q[19];
rz(1.2676801997868292) q[19];