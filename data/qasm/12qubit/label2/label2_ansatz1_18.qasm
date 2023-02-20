OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.1101978566075035) q[0];
rz(2.3856533640446744) q[0];
ry(1.6247299054643838) q[1];
rz(2.2422987001127987) q[1];
ry(1.4808868396655832) q[2];
rz(-1.7377518216636645) q[2];
ry(-7.048299040235406e-05) q[3];
rz(1.612101029188401) q[3];
ry(1.5719793308545706) q[4];
rz(1.1093235172703841) q[4];
ry(1.5720030529411053) q[5];
rz(0.25497532135327994) q[5];
ry(3.1413433383396607) q[6];
rz(-3.018162667940791) q[6];
ry(0.0031955571312159824) q[7];
rz(3.103357313029406) q[7];
ry(0.1179183614255237) q[8];
rz(-2.6175214956241852) q[8];
ry(1.2886805233046) q[9];
rz(1.2493979621611455) q[9];
ry(3.1407560980932803) q[10];
rz(-2.4981816185778) q[10];
ry(1.5175985289168006) q[11];
rz(-2.891242795603739) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.6727342321111819) q[0];
rz(2.4818997339714484) q[0];
ry(-1.2407887408222615) q[1];
rz(-1.3832299061066025) q[1];
ry(-2.1106584623981193) q[2];
rz(1.0585506390558985) q[2];
ry(-2.25515972820087) q[3];
rz(1.318929458251454) q[3];
ry(1.287387223403308) q[4];
rz(-2.8057932829675334) q[4];
ry(0.5811175183795728) q[5];
rz(1.738445327262642) q[5];
ry(-1.5921509696054565) q[6];
rz(-0.8291384025964402) q[6];
ry(-0.003053854452708967) q[7];
rz(-0.6214527863121904) q[7];
ry(-1.6451607898738705) q[8];
rz(1.449027571972617) q[8];
ry(1.3893511498074682) q[9];
rz(0.24069108518954255) q[9];
ry(-2.753501550514004) q[10];
rz(-1.6711654254053883) q[10];
ry(1.5450373522014853) q[11];
rz(-3.029553039897161) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.004337360665538946) q[0];
rz(1.2696396635891278) q[0];
ry(3.1205107393352542) q[1];
rz(-2.2200399378868525) q[1];
ry(0.810518077150166) q[2];
rz(-2.883104418649661) q[2];
ry(3.1414823256614195) q[3];
rz(-1.772286547949144) q[3];
ry(-1.036186396625329) q[4];
rz(3.132557444419392) q[4];
ry(-0.004382902150623202) q[5];
rz(0.21917340457303938) q[5];
ry(-3.1389104888261183) q[6];
rz(1.884050223025704) q[6];
ry(-0.02417195669649316) q[7];
rz(2.4316592027104873) q[7];
ry(-1.0903181238848392) q[8];
rz(0.3411722495230798) q[8];
ry(-0.020120083094709255) q[9];
rz(-0.053709705375139194) q[9];
ry(-2.8825751065015033) q[10];
rz(-2.0439864451011487) q[10];
ry(1.389591766841325) q[11];
rz(2.105509932959129) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5471448374319166) q[0];
rz(0.6910112278994326) q[0];
ry(-1.6631569690037906) q[1];
rz(-1.522302130616473) q[1];
ry(2.622003655674625) q[2];
rz(2.5111408742881443) q[2];
ry(3.141151610549897) q[3];
rz(0.8556463371909963) q[3];
ry(1.4515074644739836) q[4];
rz(3.1283704441176226) q[4];
ry(0.0039528463307238314) q[5];
rz(-0.15244823835984267) q[5];
ry(-0.023434134119653994) q[6];
rz(0.8832130773077402) q[6];
ry(-0.00010122263616692753) q[7];
rz(-2.3733413984680762) q[7];
ry(0.38121772379799673) q[8];
rz(-1.834257841254004) q[8];
ry(0.8364129455790149) q[9];
rz(-0.8371335167290983) q[9];
ry(0.09862141350579279) q[10];
rz(1.8707596709200287) q[10];
ry(2.743727368425872) q[11];
rz(-2.0945751849057763) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.896909398051166) q[0];
rz(-0.05740727783659495) q[0];
ry(0.5369426370824236) q[1];
rz(0.8151346488173666) q[1];
ry(3.1309853955946174) q[2];
rz(1.1307015172532777) q[2];
ry(-1.9182311717233513) q[3];
rz(3.1415585345453367) q[3];
ry(2.1270123264516014) q[4];
rz(1.703991153358178) q[4];
ry(-2.9752704461134716) q[5];
rz(-2.4404968197541854) q[5];
ry(-2.271563093884124) q[6];
rz(0.1772296953353143) q[6];
ry(-2.5265157249233843) q[7];
rz(0.18696392092892467) q[7];
ry(0.17803346317419155) q[8];
rz(1.4070573909730604) q[8];
ry(-0.0036336892438475483) q[9];
rz(-1.735707050590344) q[9];
ry(-2.921566635110977) q[10];
rz(1.311776057272203) q[10];
ry(-0.7802900628675555) q[11];
rz(-1.9775968230268477) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.6972590877485259) q[0];
rz(1.4237200845784133) q[0];
ry(0.1513443489012745) q[1];
rz(1.1584051677366944) q[1];
ry(3.141433944839476) q[2];
rz(-0.9355610037819792) q[2];
ry(-2.439554004164719) q[3];
rz(0.0019485619699210225) q[3];
ry(-0.003745834317503416) q[4];
rz(-1.950177772959476) q[4];
ry(-3.0427579470561827) q[5];
rz(2.396134566862414) q[5];
ry(3.1409148646086043) q[6];
rz(-2.0132909319418104) q[6];
ry(0.0006139758072085617) q[7];
rz(-1.497195215065968) q[7];
ry(3.0691441789847955) q[8];
rz(0.5671832910390799) q[8];
ry(0.5319312702929402) q[9];
rz(-1.3821269746532872) q[9];
ry(0.03280434225563411) q[10];
rz(2.1800932288154193) q[10];
ry(-1.9506609696593893) q[11];
rz(2.0908611981795198) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.785517485469791) q[0];
rz(1.281932803902036) q[0];
ry(-0.17375688734653583) q[1];
rz(0.36942793955152803) q[1];
ry(1.5222394953198075) q[2];
rz(2.73707232517812) q[2];
ry(-1.227111848790143) q[3];
rz(2.7447370274000154) q[3];
ry(-0.04411372824594385) q[4];
rz(-3.117475048730807) q[4];
ry(2.087330510119237) q[5];
rz(0.22282695629242907) q[5];
ry(0.39712796904693004) q[6];
rz(-2.218380987885274) q[6];
ry(1.290179202770644) q[7];
rz(-2.9070420959535186) q[7];
ry(-0.03774249685363084) q[8];
rz(0.8675359554874955) q[8];
ry(3.1403521616012484) q[9];
rz(0.7140585015266581) q[9];
ry(2.221255774900156) q[10];
rz(-0.1620258038602203) q[10];
ry(-0.6010896769736549) q[11];
rz(2.352096637289934) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9688933232558243) q[0];
rz(-2.2536207992012485) q[0];
ry(0.14803691751596837) q[1];
rz(3.1376204971560036) q[1];
ry(-2.895880053623624) q[2];
rz(1.2010958067009847) q[2];
ry(-0.15555664033760747) q[3];
rz(-2.9810820828068008) q[3];
ry(0.7569807410913754) q[4];
rz(2.50876474087231) q[4];
ry(2.89631935852295) q[5];
rz(0.5768518844314965) q[5];
ry(0.0025483444462060818) q[6];
rz(2.448125793910341) q[6];
ry(3.1411952048189598) q[7];
rz(2.845077170271219) q[7];
ry(-1.5556200929871329) q[8];
rz(1.1031385699760072) q[8];
ry(3.1088350242369214) q[9];
rz(1.8162074975269749) q[9];
ry(0.14287744857210027) q[10];
rz(-0.17184359209653247) q[10];
ry(-2.822527777453642) q[11];
rz(2.2171422856463687) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.6923684165981572) q[0];
rz(-1.2275062727912163) q[0];
ry(-0.025081626121760795) q[1];
rz(2.477790232892813) q[1];
ry(-0.1309811622074254) q[2];
rz(-0.6060640884239749) q[2];
ry(0.005566634159152722) q[3];
rz(2.255977918929374) q[3];
ry(-3.12218431492608) q[4];
rz(1.4334195733630875) q[4];
ry(2.1183581542663696) q[5];
rz(-0.084241258075469) q[5];
ry(2.4199672701502335) q[6];
rz(1.0860076161693133) q[6];
ry(1.337652480176418) q[7];
rz(-0.2603735646594503) q[7];
ry(-2.49242892287959) q[8];
rz(0.1438503083595705) q[8];
ry(1.3063653730690552) q[9];
rz(2.9462202725449163) q[9];
ry(-1.977172065116604) q[10];
rz(-0.42416162382061007) q[10];
ry(2.4152164331942516) q[11];
rz(-0.9532107174002533) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.729527960460045) q[0];
rz(1.438869975942382) q[0];
ry(-0.4461346285279634) q[1];
rz(2.4428012200160647) q[1];
ry(1.5774778039140882) q[2];
rz(2.793617190300454) q[2];
ry(-0.49855770688949314) q[3];
rz(-0.5974791069606235) q[3];
ry(-2.434686273682902) q[4];
rz(0.7680788875859976) q[4];
ry(-2.7693367841168293) q[5];
rz(-1.786870720626979) q[5];
ry(0.022432152616448545) q[6];
rz(-2.6632473227213187) q[6];
ry(0.004943677145632883) q[7];
rz(1.7108465847378547) q[7];
ry(-3.107307505986359) q[8];
rz(-0.02168752096267568) q[8];
ry(1.638453689499e-05) q[9];
rz(1.910279542858398) q[9];
ry(-0.08617244836716419) q[10];
rz(-1.9184735800280808) q[10];
ry(-2.9100226295527305) q[11];
rz(1.1106816439138152) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.367586646722471) q[0];
rz(1.1969127321257016) q[0];
ry(0.0020628878435493547) q[1];
rz(-2.845493921213425) q[1];
ry(-3.139205060005151) q[2];
rz(-1.3945770878346053) q[2];
ry(3.1415152587432367) q[3];
rz(-0.43997817933858263) q[3];
ry(3.0516342882206993) q[4];
rz(-2.0361339459525456) q[4];
ry(2.9270535760157856) q[5];
rz(0.9670247227119751) q[5];
ry(-2.982580494960059) q[6];
rz(1.3115370178580779) q[6];
ry(0.2125546102780764) q[7];
rz(0.7202992394236521) q[7];
ry(-0.7151965105302294) q[8];
rz(1.4362848514842268) q[8];
ry(1.8155029664358748) q[9];
rz(0.7011745067907463) q[9];
ry(2.8594389489881102) q[10];
rz(2.1598849418808874) q[10];
ry(2.5154374484336772) q[11];
rz(-1.5286513166397728) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.599446761988955) q[0];
rz(-1.1886739370225488) q[0];
ry(-2.114444066755273) q[1];
rz(-1.3591511010759898) q[1];
ry(-2.1511490901340613) q[2];
rz(2.4357614969355335) q[2];
ry(1.9314021750862866) q[3];
rz(-0.0929331584710953) q[3];
ry(2.835729941510323) q[4];
rz(2.8155917394192453) q[4];
ry(-2.7892203449897846) q[5];
rz(-2.0857362678085893) q[5];
ry(3.123106734560766) q[6];
rz(0.8917324313406729) q[6];
ry(-0.6176582657558054) q[7];
rz(-1.1909062377348019) q[7];
ry(0.24900054182456033) q[8];
rz(1.304133849383889) q[8];
ry(2.5433099649048136) q[9];
rz(-2.9154373211013933) q[9];
ry(0.04453542232546592) q[10];
rz(-2.574667264278586) q[10];
ry(-2.8620112912112807) q[11];
rz(1.0771497887786772) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.091776708662718) q[0];
rz(-1.3211734291968265) q[0];
ry(-3.099857805837065) q[1];
rz(2.668829174894478) q[1];
ry(3.130213189121571) q[2];
rz(0.1211397669436504) q[2];
ry(-0.005703483980369128) q[3];
rz(0.28512387796475686) q[3];
ry(0.02719743483048198) q[4];
rz(-2.3945070728110807) q[4];
ry(-0.07969640204367767) q[5];
rz(0.9867451143140447) q[5];
ry(-3.099661596841823) q[6];
rz(-3.13643888888083) q[6];
ry(0.02123567543291246) q[7];
rz(1.8602501682330788) q[7];
ry(-0.007616885160478546) q[8];
rz(2.093499757425777) q[8];
ry(-2.225938174113305) q[9];
rz(-0.7828747652655422) q[9];
ry(-0.45669219968131913) q[10];
rz(-2.7398106895931855) q[10];
ry(-1.7254188198724214) q[11];
rz(-2.224799473429713) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.6866450361354475) q[0];
rz(1.3047328420977022) q[0];
ry(-0.17457602089639984) q[1];
rz(-0.9318635331474211) q[1];
ry(-3.029839479206279) q[2];
rz(2.3068712718604245) q[2];
ry(-1.6727363685145085) q[3];
rz(-2.6971226869767024) q[3];
ry(0.3127939683820807) q[4];
rz(1.679256552281955) q[4];
ry(-1.9964365632425718) q[5];
rz(-2.949489839526457) q[5];
ry(1.4182626005899521) q[6];
rz(-1.5907274735386887) q[6];
ry(-0.31558719056992346) q[7];
rz(-2.641932156108198) q[7];
ry(1.3530367250076747) q[8];
rz(0.6817352956041001) q[8];
ry(-3.09008145661698) q[9];
rz(1.770660749180468) q[9];
ry(2.14526645165164) q[10];
rz(-0.7387794269462149) q[10];
ry(2.796347206417935) q[11];
rz(-0.2583941083125074) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.2247144926454023) q[0];
rz(0.8071227263811331) q[0];
ry(-3.100697746338616) q[1];
rz(-2.910311766857046) q[1];
ry(-0.05411616787396712) q[2];
rz(0.14281638353626036) q[2];
ry(3.141461893484346) q[3];
rz(0.30334737429607816) q[3];
ry(-2.096465367258885) q[4];
rz(3.0573558512760592) q[4];
ry(0.10456608839115096) q[5];
rz(0.8178394289116264) q[5];
ry(-3.092316260540239) q[6];
rz(-1.5594585375050092) q[6];
ry(-0.047750045599318724) q[7];
rz(-0.5792830457006113) q[7];
ry(-0.015193468648448949) q[8];
rz(0.689687325910028) q[8];
ry(-0.00035893596065150746) q[9];
rz(0.7069953003854156) q[9];
ry(-0.06222846677023863) q[10];
rz(2.540931788979284) q[10];
ry(3.022984377691814) q[11];
rz(-0.25719169251145235) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.8325153834978956) q[0];
rz(2.3354745169452134) q[0];
ry(-0.8962648621348137) q[1];
rz(-3.0005785560941844) q[1];
ry(-0.9551877323091122) q[2];
rz(1.4469452540169738) q[2];
ry(0.06624279878049766) q[3];
rz(1.065819313622796) q[3];
ry(3.1169460895303978) q[4];
rz(-0.1959622968813708) q[4];
ry(0.012202351967305047) q[5];
rz(-2.6370624261049893) q[5];
ry(-2.32069699849556) q[6];
rz(2.071945772331766) q[6];
ry(-2.6766986851807464) q[7];
rz(-2.0197172867540965) q[7];
ry(-2.9020786565378773) q[8];
rz(1.1581584002629433) q[8];
ry(3.1151380341489245) q[9];
rz(2.2717760003612613) q[9];
ry(-2.5636435299919187) q[10];
rz(0.5586093590949286) q[10];
ry(0.5519086711949114) q[11];
rz(2.2243767087322164) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.023520541917936) q[0];
rz(1.4052110920916285) q[0];
ry(1.2637864878613372) q[1];
rz(2.442921194604662) q[1];
ry(0.03951807357662569) q[2];
rz(2.4560708343384294) q[2];
ry(-3.1057373535037835) q[3];
rz(2.0980674364759335) q[3];
ry(0.636191084330058) q[4];
rz(0.9309246516789528) q[4];
ry(0.7570711195977927) q[5];
rz(0.46225973649795454) q[5];
ry(-3.103502312630122) q[6];
rz(1.9926154519763117) q[6];
ry(-3.0904937759284103) q[7];
rz(-0.37196417137431403) q[7];
ry(-0.024731083445836965) q[8];
rz(3.0601466604770096) q[8];
ry(-2.9891144711475874) q[9];
rz(2.888707810965402) q[9];
ry(3.0704597467856303) q[10];
rz(2.3277332262542054) q[10];
ry(-0.5311699474140027) q[11];
rz(0.059846181430782616) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.34037215937119747) q[0];
rz(2.670668482400968) q[0];
ry(-0.9757640880884324) q[1];
rz(2.1895666097100603) q[1];
ry(-0.1409647432292811) q[2];
rz(0.7509704328843548) q[2];
ry(-2.7872486545941997) q[3];
rz(0.6289669409135304) q[3];
ry(3.131771265878092) q[4];
rz(-1.5097721236805837) q[4];
ry(0.006373004926974346) q[5];
rz(-0.4095701563016157) q[5];
ry(-0.016880276818818984) q[6];
rz(0.13354374601786662) q[6];
ry(-0.8643830577520866) q[7];
rz(1.9615038980084796) q[7];
ry(-2.90760874366101) q[8];
rz(0.17765331496696213) q[8];
ry(3.093518801645541) q[9];
rz(-0.9724356666156402) q[9];
ry(-1.1283271912942068) q[10];
rz(-1.054949313243924) q[10];
ry(2.3284060321417077) q[11];
rz(-3.1345986819467395) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.106104071295294) q[0];
rz(1.665633537106175) q[0];
ry(1.1994700936175373) q[1];
rz(0.3666347077393704) q[1];
ry(-0.013056001711281127) q[2];
rz(0.6674668991783796) q[2];
ry(-3.0464560624814063) q[3];
rz(0.227622757412575) q[3];
ry(2.9893664637944832) q[4];
rz(-2.3976821745844794) q[4];
ry(-2.339574161693704) q[5];
rz(2.9813920453963245) q[5];
ry(-0.9594642023643258) q[6];
rz(0.2435080890235281) q[6];
ry(0.13423373849051734) q[7];
rz(1.8403057737264241) q[7];
ry(-0.015108597457526093) q[8];
rz(-2.999584239257187) q[8];
ry(-0.029170748913795366) q[9];
rz(-2.292686080138224) q[9];
ry(2.72746268955794) q[10];
rz(1.1253324210656963) q[10];
ry(-1.9483995318319627) q[11];
rz(-0.9431614843875777) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.0329263180214427) q[0];
rz(-3.0024609025887816) q[0];
ry(-0.7356724955119969) q[1];
rz(-0.8454801007592261) q[1];
ry(-3.1122347712461216) q[2];
rz(1.995780541784681) q[2];
ry(0.4429795457399187) q[3];
rz(0.8935694243819361) q[3];
ry(-0.2975638118661825) q[4];
rz(3.092792262965196) q[4];
ry(3.1295972862771992) q[5];
rz(1.5423774753322244) q[5];
ry(-3.0189299199203266) q[6];
rz(0.2353809403945883) q[6];
ry(3.1326191634087825) q[7];
rz(1.6115089804524925) q[7];
ry(-3.0437884699621) q[8];
rz(-2.004127909032694) q[8];
ry(-3.1314435930581834) q[9];
rz(-2.693387786206883) q[9];
ry(-1.3527301859683543) q[10];
rz(-0.9867361953849599) q[10];
ry(-2.9170331331668944) q[11];
rz(0.10405845887488763) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.099063627163319) q[0];
rz(1.1932029069467571) q[0];
ry(2.53588044851353) q[1];
rz(2.049373134611347) q[1];
ry(-0.007606535370864087) q[2];
rz(-2.9887072674583757) q[2];
ry(-3.0613505814265807) q[3];
rz(2.4906704841103124) q[3];
ry(-0.47798408385234836) q[4];
rz(-2.5656292507508183) q[4];
ry(0.10377261451572295) q[5];
rz(1.2521345771709842) q[5];
ry(-2.1793508742588488) q[6];
rz(-0.7632018429510617) q[6];
ry(-0.15403122591738438) q[7];
rz(2.5923731486532193) q[7];
ry(0.054049127419230736) q[8];
rz(-0.5708582006979439) q[8];
ry(-0.12645456467763075) q[9];
rz(2.8443211281470764) q[9];
ry(0.8982828333001833) q[10];
rz(0.881861887603792) q[10];
ry(1.554681457505608) q[11];
rz(0.2686129032263639) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.0407849324884686) q[0];
rz(2.9492317123802034) q[0];
ry(-0.6964539342885425) q[1];
rz(0.4760020894627095) q[1];
ry(-2.107587483370711) q[2];
rz(-2.8231904342267886) q[2];
ry(0.9056250417132121) q[3];
rz(-3.1207453706353983) q[3];
ry(0.4222893018266704) q[4];
rz(0.6610122436975228) q[4];
ry(-1.9148517958821998) q[5];
rz(1.235166174855637) q[5];
ry(2.62652983922028) q[6];
rz(2.1642543438021487) q[6];
ry(-1.7754073878601888) q[7];
rz(0.47401716727662496) q[7];
ry(-1.492654104253361) q[8];
rz(1.9622776241390243) q[8];
ry(-1.30832400124261) q[9];
rz(2.1187946324107916) q[9];
ry(-0.5554745021824273) q[10];
rz(-2.565505834107941) q[10];
ry(-1.0383788275030312) q[11];
rz(0.012308418729235271) q[11];