OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.0032375509670765977) q[0];
rz(2.336913882438059) q[0];
ry(1.446891434287826) q[1];
rz(-0.04081770526370601) q[1];
ry(-5.260206814059889e-06) q[2];
rz(1.293885451250752) q[2];
ry(1.5724822973701924) q[3];
rz(1.8964851757793117) q[3];
ry(-1.5674948484761067) q[4];
rz(-1.614426896348463) q[4];
ry(-1.7761381152517837) q[5];
rz(-2.0928224563348787) q[5];
ry(-3.1415637472311824) q[6];
rz(1.5066135998478072) q[6];
ry(-1.570803659074354) q[7];
rz(1.5708376957522736) q[7];
ry(1.4736528157944342) q[8];
rz(4.0622180375548284e-05) q[8];
ry(-3.141587528875034) q[9];
rz(0.8934908517105011) q[9];
ry(-3.1415162382168593) q[10];
rz(1.1351911901396405) q[10];
ry(-1.285344161991346) q[11];
rz(1.3874174418885357) q[11];
ry(2.197721221021853) q[12];
rz(2.9521276183765646) q[12];
ry(-3.101943175527299) q[13];
rz(2.821641658800292) q[13];
ry(1.6691616814945576e-05) q[14];
rz(1.006877294112771) q[14];
ry(0.0001905371558841291) q[15];
rz(2.4928463437701396) q[15];
ry(-2.5725238247036244) q[16];
rz(-2.342941174880876) q[16];
ry(-3.13787198075765) q[17];
rz(-2.7843504594321904) q[17];
ry(-0.004806346302933306) q[18];
rz(-0.8135813665327699) q[18];
ry(0.0795321504050071) q[19];
rz(0.6360859722315899) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.5709341889861004) q[0];
rz(0.8811843797195102) q[0];
ry(0.10285923073997338) q[1];
rz(-3.1058521869103566) q[1];
ry(-1.5660091813385693) q[2];
rz(0.8059011808091079) q[2];
ry(0.0003283602192672862) q[3];
rz(-0.34640059140152335) q[3];
ry(-2.4787812618192118) q[4];
rz(-0.050414812732617165) q[4];
ry(-1.5664044056743143) q[5];
rz(-2.8593047229463764e-05) q[5];
ry(-1.5707907252361049) q[6];
rz(-2.3629390302972544) q[6];
ry(1.6471356002509516) q[7];
rz(-2.6012677927848893) q[7];
ry(-1.667944635150083) q[8];
rz(-2.441758109008396) q[8];
ry(-6.0465375014253385e-06) q[9];
rz(0.3148992795170581) q[9];
ry(1.5132418154097629) q[10];
rz(-1.5300637853181396) q[10];
ry(1.02285146029365) q[11];
rz(2.6605653645458713) q[11];
ry(-2.504067805903014) q[12];
rz(-1.9645597674274582) q[12];
ry(-0.007342353338844809) q[13];
rz(-1.7893391826345073) q[13];
ry(-1.5712720379018998) q[14];
rz(0.6180706246941544) q[14];
ry(0.003683786631222929) q[15];
rz(-0.016599340138062434) q[15];
ry(-2.887865641690009) q[16];
rz(2.141671807777813) q[16];
ry(1.434241263941279) q[17];
rz(0.1004185830196027) q[17];
ry(-0.014628344555344796) q[18];
rz(-0.25961579419169006) q[18];
ry(2.55003188252012) q[19];
rz(0.029884430122379124) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(3.1408063662411236) q[0];
rz(0.9743642700896148) q[0];
ry(-1.5734149113253162) q[1];
rz(-1.4670921363663347) q[1];
ry(3.141544312429259) q[2];
rz(0.6957123182781482) q[2];
ry(3.141582136912305) q[3];
rz(-1.575047669493231) q[3];
ry(-3.0953475477578007) q[4];
rz(0.14120189732534882) q[4];
ry(0.5775406170544956) q[5];
rz(-3.1415620647089435) q[5];
ry(3.141485198150923) q[6];
rz(2.3494590631424255) q[6];
ry(3.141445604293325) q[7];
rz(0.3735545364822456) q[7];
ry(1.5811140633058927) q[8];
rz(-1.433963330049389) q[8];
ry(-3.132866416324643) q[9];
rz(2.819483162239938) q[9];
ry(-2.0676179477870527) q[10];
rz(0.3493687258145387) q[10];
ry(-3.137566629934489) q[11];
rz(-0.7342449977736925) q[11];
ry(-1.0585963117648587e-05) q[12];
rz(-2.919254737651716) q[12];
ry(2.705062786795931) q[13];
rz(1.145905369657326) q[13];
ry(9.07515291608263e-05) q[14];
rz(-0.6681120327497793) q[14];
ry(1.5708872989631555) q[15];
rz(2.535465550840641) q[15];
ry(-0.001040407031963241) q[16];
rz(2.0797145117080182) q[16];
ry(-0.009885920414050453) q[17];
rz(3.0428204032383843) q[17];
ry(3.1374933583996536) q[18];
rz(0.4699897630635564) q[18];
ry(0.03864115813218763) q[19];
rz(-3.1118750878049064) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(3.016092913946204) q[0];
rz(1.7672952098358072) q[0];
ry(1.5807418547740175) q[1];
rz(-3.012618189487922) q[1];
ry(0.004825172556891784) q[2];
rz(-1.287464585816138) q[2];
ry(0.19387543956108022) q[3];
rz(-0.3706419796789025) q[3];
ry(0.3460210995359066) q[4];
rz(2.947487086798612) q[4];
ry(-1.5751767081008143) q[5];
rz(-0.015243704548656643) q[5];
ry(1.5720887810467734) q[6];
rz(-0.5742052325725063) q[6];
ry(-1.8691920246326232e-05) q[7];
rz(1.7339019851275534) q[7];
ry(-3.141426544266824) q[8];
rz(-1.234413482074403) q[8];
ry(3.14158932759938) q[9];
rz(2.742623621325069) q[9];
ry(2.1832710049082173) q[10];
rz(-1.5044206156215232) q[10];
ry(-3.1069214801299085) q[11];
rz(0.34894542204587164) q[11];
ry(3.109210670282712) q[12];
rz(0.3803516337130022) q[12];
ry(-1.1131861056057348e-05) q[13];
rz(-1.1597819524123452) q[13];
ry(-1.6215376198608658) q[14];
rz(-2.5234356876332322) q[14];
ry(3.141100434827491) q[15];
rz(-3.0354174725576955) q[15];
ry(-3.072954479209563) q[16];
rz(0.46669660403977803) q[16];
ry(-1.5716782289285696) q[17];
rz(1.6096310277413357) q[17];
ry(-0.0017089263915377074) q[18];
rz(-0.2928144030544427) q[18];
ry(-1.9121212025215906) q[19];
rz(-3.123889155015683) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.570382565081029) q[0];
rz(-0.5674842498802459) q[0];
ry(0.00546655219998371) q[1];
rz(-0.3187119945136301) q[1];
ry(3.0941053655890824) q[2];
rz(-3.062709212821392) q[2];
ry(0.0001365463458515039) q[3];
rz(-2.7929898112171743) q[3];
ry(1.573698946878956) q[4];
rz(1.632933408853221) q[4];
ry(-0.17610693643637187) q[5];
rz(2.0437988777799756) q[5];
ry(7.232176294826392e-05) q[6];
rz(2.4477211590201686) q[6];
ry(-1.570927962958) q[7];
rz(1.5325800810465964) q[7];
ry(-0.014347859603931036) q[8];
rz(-1.2660359363550515) q[8];
ry(-3.071942350200397) q[9];
rz(-1.6307725498464618) q[9];
ry(3.108000965067509) q[10];
rz(2.4553140173406813) q[10];
ry(-0.0918779024488412) q[11];
rz(1.4626668904958415) q[11];
ry(3.183716176202723e-05) q[12];
rz(-1.61114764180462) q[12];
ry(0.6516662345715954) q[13];
rz(-2.097848953960402) q[13];
ry(-3.1407804232742635) q[14];
rz(1.3922598332413305) q[14];
ry(-1.0202158695871661) q[15];
rz(-2.5610177483306766) q[15];
ry(-2.4123842282090524) q[16];
rz(-0.45139425770516345) q[16];
ry(3.1106729377256688) q[17];
rz(-3.13578766735321) q[17];
ry(-2.773887749901465) q[18];
rz(-0.01231667897059988) q[18];
ry(1.5668915424574044) q[19];
rz(2.530898414138941) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.0058892397950829995) q[0];
rz(2.710013120485906) q[0];
ry(-1.6991214909572463) q[1];
rz(-0.6149149895553718) q[1];
ry(-2.449350817381536) q[2];
rz(2.927160591799336) q[2];
ry(1.379841378887291) q[3];
rz(0.40947947243926297) q[3];
ry(-1.5369696797622128) q[4];
rz(0.3382087566676537) q[4];
ry(-3.125580522380863) q[5];
rz(-2.1499093336984645) q[5];
ry(-3.1415443670150003) q[6];
rz(1.2549158227112547) q[6];
ry(3.141573243716845) q[7];
rz(-0.07526666546846685) q[7];
ry(0.00013086150498229005) q[8];
rz(0.3843255685120652) q[8];
ry(3.1415476885798355) q[9];
rz(1.52016701977041) q[9];
ry(-0.27490564473698326) q[10];
rz(1.3521145492330795) q[10];
ry(-3.12725028526484) q[11];
rz(-1.3997701201831918) q[11];
ry(-3.1342587982531294) q[12];
rz(-1.3968893936520628) q[12];
ry(3.1414318608353735) q[13];
rz(1.8822563159727137) q[13];
ry(0.0978892842580565) q[14];
rz(1.3074436732997345) q[14];
ry(-3.1414402818925002) q[15];
rz(0.2471097583940205) q[15];
ry(0.021399343517109592) q[16];
rz(0.43155224966973105) q[16];
ry(3.141452110656942) q[17];
rz(-2.9465321674703833) q[17];
ry(1.5705238964691315) q[18];
rz(1.6734948761414439) q[18];
ry(-3.1324805553768837) q[19];
rz(2.436842118443853) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(3.1415098980124405) q[0];
rz(2.7513515840165703) q[0];
ry(-1.5984165501727228) q[1];
rz(2.503543910373491) q[1];
ry(-3.0928622743877385) q[2];
rz(-0.25075593708464444) q[2];
ry(3.1403547256985185) q[3];
rz(0.06379894356715939) q[3];
ry(-0.056112204450132985) q[4];
rz(-0.5765368425150926) q[4];
ry(-0.19236787822003176) q[5];
rz(2.168469891305537) q[5];
ry(-1.359431116121357e-06) q[6];
rz(-1.335841748881534) q[6];
ry(0.003688023618118486) q[7];
rz(-0.5281925844368928) q[7];
ry(-1.5463229980955546) q[8];
rz(-3.1207026402576523) q[8];
ry(0.07836983949335607) q[9];
rz(-0.6114437613212583) q[9];
ry(-0.5887853569716635) q[10];
rz(1.537946019690936) q[10];
ry(-2.0373835140686367) q[11];
rz(-1.542322482952509) q[11];
ry(-3.1415560055614273) q[12];
rz(1.4303476865501494) q[12];
ry(0.6298921679057583) q[13];
rz(0.47200571784010137) q[13];
ry(-0.0009420335411632794) q[14];
rz(-2.87324073926398) q[14];
ry(-2.5616273961557914) q[15];
rz(2.4530665111841334) q[15];
ry(1.596320825195817) q[16];
rz(-2.897608851201386) q[16];
ry(0.03264339359678365) q[17];
rz(-1.0393755040494932) q[17];
ry(0.8631350879471196) q[18];
rz(-0.7878640722945445) q[18];
ry(-1.5906182087819514) q[19];
rz(-2.85248105301402) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-2.599269410835025) q[0];
rz(1.4498711614220232) q[0];
ry(3.048124057518335) q[1];
rz(2.9315610855487106) q[1];
ry(2.269879795666082) q[2];
rz(-2.684157364741669) q[2];
ry(-7.706810419526699e-05) q[3];
rz(-3.003298868901684) q[3];
ry(0.07016226340979692) q[4];
rz(0.5226242480240267) q[4];
ry(3.0839673135056653) q[5];
rz(1.582631080537257) q[5];
ry(-2.1220541724833937e-05) q[6];
rz(1.0750210146560706) q[6];
ry(3.1415685621882607) q[7];
rz(1.9839006628053344) q[7];
ry(-1.570755153393621) q[8];
rz(1.5707630019340921) q[8];
ry(1.570825796505697) q[9];
rz(-1.5708519154591132) q[9];
ry(3.0783135341564014) q[10];
rz(0.7252712109639265) q[10];
ry(-1.5726298450291392) q[11];
rz(-2.9066948381997033) q[11];
ry(1.590606682768124) q[12];
rz(-0.8630125336045126) q[12];
ry(2.2929912843567055) q[13];
rz(1.5873128320383345) q[13];
ry(1.5379108842578366) q[14];
rz(-1.6432440652936888) q[14];
ry(3.141240273867478) q[15];
rz(-0.44988828039469714) q[15];
ry(3.140770414184546) q[16];
rz(2.382005844720023) q[16];
ry(-3.141119214026125) q[17];
rz(0.04294873170083946) q[17];
ry(0.19498155712044568) q[18];
rz(0.3437199794793731) q[18];
ry(-1.0462959020844318) q[19];
rz(1.6371391951479541) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(2.94219725967213e-05) q[0];
rz(-1.8593436320103098) q[0];
ry(-0.03440740050960334) q[1];
rz(2.439483498967596) q[1];
ry(-0.0016661947348565393) q[2];
rz(1.0676576226311933) q[2];
ry(-3.141046194330223) q[3];
rz(-2.425836788770991) q[3];
ry(0.041991162385413006) q[4];
rz(-2.476359608829687) q[4];
ry(2.7692283639418367) q[5];
rz(0.46750005557295465) q[5];
ry(2.0718389497353936e-05) q[6];
rz(-2.4329301776284638) q[6];
ry(-9.526059162121704e-06) q[7];
rz(1.8730728739659481) q[7];
ry(-1.5704827732355788) q[8];
rz(-1.571435931868617) q[8];
ry(-1.5706262546982748) q[9];
rz(1.5706769927309399) q[9];
ry(-3.957865382288252e-05) q[10];
rz(-1.0136435812722375) q[10];
ry(2.8616993255070607e-06) q[11];
rz(2.098111210100022) q[11];
ry(5.10729353031536e-06) q[12];
rz(1.20955607304428) q[12];
ry(2.9778369295947e-05) q[13];
rz(-1.5864965452780964) q[13];
ry(0.6250140023128783) q[14];
rz(-1.5438598193880058) q[14];
ry(-3.1267239430569482) q[15];
rz(-0.32929335434593376) q[15];
ry(-2.9960085763642486) q[16];
rz(-1.7512595996787983) q[16];
ry(-0.1296230429948082) q[17];
rz(1.6754299898651923) q[17];
ry(-0.7628459155950438) q[18];
rz(2.0285210585705165) q[18];
ry(-0.13884395894529789) q[19];
rz(-1.361553087930667) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-2.0777448571551753) q[0];
rz(1.0140284612591497) q[0];
ry(-1.4917150901750933) q[1];
rz(1.1522723246738993) q[1];
ry(2.7983480150726447) q[2];
rz(3.1178601928212806) q[2];
ry(3.1414438595305443) q[3];
rz(-1.3459455246027296) q[3];
ry(3.0832969905961485) q[4];
rz(0.9744962121256178) q[4];
ry(1.5402951479809475) q[5];
rz(-1.7309127966868922) q[5];
ry(3.4287312007812668e-06) q[6];
rz(3.0336428221697) q[6];
ry(-3.1415730539067908) q[7];
rz(-2.1654871069038357) q[7];
ry(1.5704625630534463) q[8];
rz(-2.199144311961322) q[8];
ry(-1.5701767987248418) q[9];
rz(-2.234502050190561) q[9];
ry(2.9001556023926254) q[10];
rz(-2.9600274238792488) q[10];
ry(-0.00493446987800894) q[11];
rz(-1.9790624130016143) q[11];
ry(-0.004839221293224227) q[12];
rz(0.20799884760417875) q[12];
ry(2.2899765945761072) q[13];
rz(2.2435151264499753) q[13];
ry(-0.004392425679187783) q[14];
rz(-1.5325016610128572) q[14];
ry(-3.1412254871687137) q[15];
rz(-0.2886794731582871) q[15];
ry(0.00013462820546639168) q[16];
rz(-1.5083193465891296) q[16];
ry(3.141169154060705) q[17];
rz(0.09373203013142196) q[17];
ry(-0.6798063478371635) q[18];
rz(2.628982381346357) q[18];
ry(2.6589211779961905) q[19];
rz(-0.7777449256079407) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.00012923145583165763) q[0];
rz(0.6475039407436718) q[0];
ry(-0.8396584771423767) q[1];
rz(0.009370943244427755) q[1];
ry(-1.5625129102271968) q[2];
rz(0.0005097790339409114) q[2];
ry(-3.1415599138218764) q[3];
rz(2.5802973000172402) q[3];
ry(-1.5710987427435823) q[4];
rz(-3.139757996014139) q[4];
ry(1.600584221337185) q[5];
rz(0.23721058765675185) q[5];
ry(-0.04688207834982362) q[6];
rz(-2.129019076170473) q[6];
ry(0.17554070369929783) q[7];
rz(-0.5149554989805916) q[7];
ry(0.0010929269951799278) q[8];
rz(3.1317341660277105) q[8];
ry(2.8732004945672687) q[9];
rz(0.6078125449587829) q[9];
ry(3.061782385995297) q[10];
rz(-2.9584944499273655) q[10];
ry(3.1385301084598742) q[11];
rz(-0.5414530187409349) q[11];
ry(0.00012980748501466663) q[12];
rz(-1.5104603435666035) q[12];
ry(0.0018234706767275505) q[13];
rz(0.6806929723407888) q[13];
ry(-2.2395318872591585) q[14];
rz(2.244710249828376) q[14];
ry(1.5610083618313517) q[15];
rz(2.890727253548789) q[15];
ry(3.0861838357938605) q[16];
rz(-0.5480249053371983) q[16];
ry(3.104542292408521) q[17];
rz(0.23302039216519876) q[17];
ry(2.206399160171017) q[18];
rz(-3.0828300992236426) q[18];
ry(0.03221962279590506) q[19];
rz(0.38118105073904085) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.8471290854276092) q[0];
rz(-1.99214208749152) q[0];
ry(-1.3006058769760724) q[1];
rz(1.0592817652617113) q[1];
ry(1.5706170917575346) q[2];
rz(-2.5142947927390065) q[2];
ry(3.1415120122861433) q[3];
rz(2.0761534192781017) q[3];
ry(-1.5642311589195668) q[4];
rz(-3.138904553084823) q[4];
ry(1.5752636228809893) q[5];
rz(-1.4705881816661934) q[5];
ry(-3.141571432808279) q[6];
rz(-0.3050928472080357) q[6];
ry(-0.00017645026958967793) q[7];
rz(1.2499964431735702) q[7];
ry(0.0007868369711454439) q[8];
rz(2.0361057219971537) q[8];
ry(-0.0004595868002708224) q[9];
rz(1.875994118759663) q[9];
ry(-0.2416561355764273) q[10];
rz(2.747696203102283) q[10];
ry(-3.138916084076096) q[11];
rz(2.1847828097840636) q[11];
ry(-0.035805895995192606) q[12];
rz(-2.6745415228447293) q[12];
ry(1.5817212362375628) q[13];
rz(1.27019014359443) q[13];
ry(3.1415244799880244) q[14];
rz(2.601608760747149) q[14];
ry(7.48101623990749e-05) q[15];
rz(0.42364702121115183) q[15];
ry(0.0003377898614349561) q[16];
rz(0.8816240486422622) q[16];
ry(-0.04516700812139448) q[17];
rz(-0.5999628918654913) q[17];
ry(-0.5939386386167944) q[18];
rz(1.0738070955835601) q[18];
ry(-0.5413418267697002) q[19];
rz(-0.4614562669744626) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.41964731828309026) q[0];
rz(1.3984610787985687) q[0];
ry(-2.4266586675319353) q[1];
rz(-0.47609288586125115) q[1];
ry(-0.07010291687923509) q[2];
rz(-1.6767691596831276) q[2];
ry(-3.1414556205558672) q[3];
rz(-2.133234481829641) q[3];
ry(1.5745544838379408) q[4];
rz(-0.10889580361711088) q[4];
ry(1.5676067019209305) q[5];
rz(1.566388737406088) q[5];
ry(-0.16413510175291626) q[6];
rz(-0.5987045114404164) q[6];
ry(-3.1415714184012873) q[7];
rz(-2.465187625401056) q[7];
ry(3.02648963824595) q[8];
rz(-0.4300795182663089) q[8];
ry(-1.4263948219754248) q[9];
rz(1.107292104433051) q[9];
ry(-0.9576855117960358) q[10];
rz(1.5250921422325208) q[10];
ry(1.570847243999422) q[11];
rz(-0.7629031396176105) q[11];
ry(-0.03969746610239078) q[12];
rz(-0.39208136011347977) q[12];
ry(3.141369531818357) q[13];
rz(-0.298213557920878) q[13];
ry(0.03313633031828661) q[14];
rz(-2.9024531513061747) q[14];
ry(0.003022613242124273) q[15];
rz(0.13249320892304725) q[15];
ry(1.6009832614261734) q[16];
rz(-2.3222337135323423) q[16];
ry(0.31166795596787095) q[17];
rz(2.153465889946255) q[17];
ry(-2.0748125942605578) q[18];
rz(1.7461259914017229) q[18];
ry(3.0872992970911457) q[19];
rz(1.7789795257531535) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.8293134232179467) q[0];
rz(-1.0365827369665137) q[0];
ry(3.1375514952945136) q[1];
rz(0.8750709788567756) q[1];
ry(3.1410488729103334) q[2];
rz(2.186572826361502) q[2];
ry(-3.1411802278616126) q[3];
rz(-2.441692206174946) q[3];
ry(-3.1411412126101705) q[4];
rz(0.79759521228769) q[4];
ry(-1.5660236724534145) q[5];
rz(-1.914864261357363) q[5];
ry(-3.141536552258594) q[6];
rz(-0.21020012094211182) q[6];
ry(-6.326529580277123e-05) q[7];
rz(1.7935619514470724) q[7];
ry(6.593167172342179e-05) q[8];
rz(1.0513899909931512) q[8];
ry(-3.1415844584352133) q[9];
rz(0.41846576912189626) q[9];
ry(0.00015134438689212715) q[10];
rz(2.330031368190678) q[10];
ry(4.2556354996210905e-05) q[11];
rz(-0.80775775254605) q[11];
ry(-3.0843258097477904) q[12];
rz(2.0122395367470043) q[12];
ry(1.5707832513377762) q[13];
rz(0.04352606657296181) q[13];
ry(-3.14158278773623) q[14];
rz(0.7372051570169124) q[14];
ry(3.1414172331900136) q[15];
rz(2.039805894545331) q[15];
ry(-0.04780411093649266) q[16];
rz(1.6545086192606722) q[16];
ry(2.6037870905553353) q[17];
rz(-0.7873330237470452) q[17];
ry(-3.1204579738056673) q[18];
rz(2.740002563971897) q[18];
ry(-2.0891854274132964) q[19];
rz(-1.78283249030686) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(2.7487001120257326) q[0];
rz(-2.37012693565939) q[0];
ry(-2.3618800564056643) q[1];
rz(-1.0790195418462405) q[1];
ry(-0.12005487470266765) q[2];
rz(-2.4143270548786857) q[2];
ry(0.5807394936687268) q[3];
rz(1.3204178359464107) q[3];
ry(1.2910716240339588) q[4];
rz(2.143722680437848) q[4];
ry(2.181398702671725) q[5];
rz(2.911026275979095) q[5];
ry(-0.2919120166257984) q[6];
rz(2.2218559771461415) q[6];
ry(1.6459193725773913) q[7];
rz(-2.376030273026831) q[7];
ry(-2.9794739765958633) q[8];
rz(-3.138824278427743) q[8];
ry(0.04984226074244649) q[9];
rz(1.678925950238069) q[9];
ry(-0.014252393344825087) q[10];
rz(-0.7284240871829569) q[10];
ry(-1.5708096975519972) q[11];
rz(-2.7044258119372238) q[11];
ry(-2.2018421792641695e-05) q[12];
rz(2.4079918103187405) q[12];
ry(1.571025951534708) q[13];
rz(1.5708118243152942) q[13];
ry(-1.0435759579126387) q[14];
rz(-1.7908625617731724) q[14];
ry(-7.0986420384092934e-06) q[15];
rz(2.5953203692582623) q[15];
ry(1.5021555185075846) q[16];
rz(0.6776051361821506) q[16];
ry(-0.004149492749131234) q[17];
rz(-1.414024726082622) q[17];
ry(-0.023615209689945797) q[18];
rz(0.6667162927767211) q[18];
ry(0.25420718394520403) q[19];
rz(-1.4973978670632428) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.6466118581501011) q[0];
rz(0.3962652701602982) q[0];
ry(-3.120809836026582) q[1];
rz(-0.749809852243968) q[1];
ry(-3.1406026767141184) q[2];
rz(-0.13576493452191854) q[2];
ry(3.1415225670376015) q[3];
rz(0.25076150384453194) q[3];
ry(-3.1412118286483968) q[4];
rz(-1.9653168694227423) q[4];
ry(-0.00019487781085825162) q[5];
rz(1.689731219644207) q[5];
ry(4.395085669628429e-06) q[6];
rz(0.8200565288240774) q[6];
ry(-3.1415787386513236) q[7];
rz(-2.5137875709862234) q[7];
ry(-3.141548934920522) q[8];
rz(-2.427751418753174) q[8];
ry(-1.7798585041362005e-06) q[9];
rz(2.786496536808322) q[9];
ry(-1.5714950737451412) q[10];
rz(3.108371908873539) q[10];
ry(0.0010430834090699526) q[11];
rz(1.992824047384918) q[11];
ry(1.0945886780473676e-05) q[12];
rz(3.086892406387125) q[12];
ry(-1.5707723878352189) q[13];
rz(0.24929343363148185) q[13];
ry(0.0006962607286471515) q[14];
rz(1.7921248775325738) q[14];
ry(0.0006251131652510544) q[15];
rz(1.9493595897933707) q[15];
ry(3.1315505115581757) q[16];
rz(-1.2821446230454079) q[16];
ry(-1.5652472213829427) q[17];
rz(0.12134934767250984) q[17];
ry(-0.0037918718932662045) q[18];
rz(0.9480297290518064) q[18];
ry(1.275840991155563) q[19];
rz(-0.4794059995684048) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.9246159726361838) q[0];
rz(-2.484156003180804) q[0];
ry(-1.2441357972407556) q[1];
rz(-0.2271173578335857) q[1];
ry(1.4558907420489904) q[2];
rz(-2.712070813253472) q[2];
ry(1.334166122668747) q[3];
rz(0.8500338290793812) q[3];
ry(0.6006765349292357) q[4];
rz(-0.14053026813042063) q[4];
ry(1.8219468118540014) q[5];
rz(2.5784136493872083) q[5];
ry(-2.992678241906694) q[6];
rz(0.565787314655191) q[6];
ry(1.3986404371353398) q[7];
rz(1.9285661997638242) q[7];
ry(0.015104304355052929) q[8];
rz(0.8867192773701702) q[8];
ry(-0.038672574528605244) q[9];
rz(-1.5541572884237118) q[9];
ry(2.462913720081555) q[10];
rz(-0.7814121359527552) q[10];
ry(3.1205688400244025) q[11];
rz(-2.4565847227240765) q[11];
ry(-0.014457486624072386) q[12];
rz(-1.965701216157318) q[12];
ry(3.0920617455302652) q[13];
rz(-2.8207606679558146) q[13];
ry(2.068795520372818) q[14];
rz(-2.2084794836514794) q[14];
ry(1.5274694916801996) q[15];
rz(-2.3125738949490136) q[15];
ry(-3.107779520546467) q[16];
rz(0.3659768824746482) q[16];
ry(-0.003433730855257931) q[17];
rz(-0.842839021985819) q[17];
ry(1.4572030109720986) q[18];
rz(-0.7814864623083012) q[18];
ry(-1.4704577729842594) q[19];
rz(2.374817723106985) q[19];