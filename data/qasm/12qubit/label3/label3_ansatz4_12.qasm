OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.566840109749888) q[0];
rz(-3.1169985337542947) q[0];
ry(1.5574411964054886) q[1];
rz(-0.004016674654488738) q[1];
ry(-0.054553303862583016) q[2];
rz(-3.0769929387229014) q[2];
ry(1.568413349103607) q[3];
rz(0.03430256051181142) q[3];
ry(2.796920107403831) q[4];
rz(-2.488137373150947) q[4];
ry(3.141400161566431) q[5];
rz(-2.9035922151389717) q[5];
ry(1.570165825159183) q[6];
rz(-3.137880904918799) q[6];
ry(1.5711337826861644) q[7];
rz(2.133415062443534) q[7];
ry(1.5710744341222806) q[8];
rz(-1.5530579097342703) q[8];
ry(-1.5709999945395459) q[9];
rz(3.0472772478947485) q[9];
ry(3.034129542278302) q[10];
rz(3.095039613912182) q[10];
ry(-1.540900797294662) q[11];
rz(1.5708080730521994) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.714132179489062) q[0];
rz(-0.6133282707131977) q[0];
ry(0.8288378314689561) q[1];
rz(-1.6994561699885598) q[1];
ry(0.6174505267559828) q[2];
rz(-0.01028001934396646) q[2];
ry(-1.0773733506163437) q[3];
rz(-2.4413463434158312e-05) q[3];
ry(-3.141561298885081) q[4];
rz(-2.4628608101649085) q[4];
ry(-0.2698491449564906) q[5];
rz(-1.5702064263856048) q[5];
ry(-1.527216533319482) q[6];
rz(0.4280861626813302) q[6];
ry(2.3457695765997926) q[7];
rz(-2.261136795934052) q[7];
ry(-1.566952610368427) q[8];
rz(-0.4132573174968046) q[8];
ry(-3.031226465942321) q[9];
rz(-2.6374425738408593) q[9];
ry(-0.006887730768236011) q[10];
rz(2.0094844983502647) q[10];
ry(1.56915457250375) q[11];
rz(-2.0492670765643384) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.10388117698834794) q[0];
rz(2.3273267096302974) q[0];
ry(1.597092042982563) q[1];
rz(-1.4038770117099633) q[1];
ry(2.906090332566786) q[2];
rz(0.3280463973904291) q[2];
ry(-1.569875967459815) q[3];
rz(1.7258441870488468) q[3];
ry(0.43382831966764623) q[4];
rz(0.0009391820849131705) q[4];
ry(5.498714430059445e-06) q[5];
rz(1.5572482373513408) q[5];
ry(-0.0015250370552069015) q[6];
rz(2.548000430027969) q[6];
ry(3.1398218127826314) q[7];
rz(-1.1172800487646697) q[7];
ry(-0.000910064362036578) q[8];
rz(-2.734018718648052) q[8];
ry(-0.003838797759286194) q[9];
rz(-2.1669224232897975) q[9];
ry(-1.6903603438321042) q[10];
rz(-2.575086328989631) q[10];
ry(0.0050268382450262585) q[11];
rz(0.7496512830028297) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.074510630112066) q[0];
rz(0.9375290157269552) q[0];
ry(0.05432456766545091) q[1];
rz(2.451407885714724) q[1];
ry(-0.008561029516022048) q[2];
rz(1.2130777215848134) q[2];
ry(3.127241192119359) q[3];
rz(-1.4184135513737806) q[3];
ry(1.571555182075581) q[4];
rz(3.1414209034900735) q[4];
ry(1.5709497932199417) q[5];
rz(1.649261335049306) q[5];
ry(3.118376884985031) q[6];
rz(1.4053533331676844) q[6];
ry(1.9724763139518382) q[7];
rz(1.4274241089860895) q[7];
ry(-3.1177183344899246) q[8];
rz(-1.469579692465297) q[8];
ry(1.6695865793773068) q[9];
rz(-1.5719561801419168) q[9];
ry(-0.0046596241382905745) q[10];
rz(2.1700903926756947) q[10];
ry(-1.2280496424382923) q[11];
rz(2.464577969951792) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5018981224347039) q[0];
rz(-1.7619448574913024) q[0];
ry(-3.000029106733362) q[1];
rz(1.7183285370161594) q[1];
ry(-1.5712993262895845) q[2];
rz(-1.2629579503184498) q[2];
ry(-1.5643352832005937) q[3];
rz(1.389557712463798) q[3];
ry(-1.5695033711324453) q[4];
rz(1.6346320814552733) q[4];
ry(1.582624092165327) q[5];
rz(-0.0011151626888272759) q[5];
ry(0.2411005208792454) q[6];
rz(-3.140919076713899) q[6];
ry(0.008600276304206842) q[7];
rz(1.1276204680726996) q[7];
ry(-1.571332209744608) q[8];
rz(-1.5800822113324067) q[8];
ry(-1.5698232391281266) q[9];
rz(-0.1161676878027542) q[9];
ry(2.2301079895721063) q[10];
rz(2.5254638314569506) q[10];
ry(2.71153264737673) q[11];
rz(-0.07534801928376035) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5741056985709176) q[0];
rz(-0.5865732583831536) q[0];
ry(-0.004313417007774767) q[1];
rz(-2.1132520399830645) q[1];
ry(-0.001494639297298724) q[2];
rz(-0.3072857608618237) q[2];
ry(3.140605106872266) q[3];
rz(1.3939045965415782) q[3];
ry(-0.0010904318488833553) q[4];
rz(0.035596000099241035) q[4];
ry(0.3258761760091433) q[5];
rz(-1.564633983072227) q[5];
ry(-1.5690324515626586) q[6];
rz(-2.5492703795053897) q[6];
ry(1.5710246305158995) q[7];
rz(2.7543035556095528) q[7];
ry(-1.569190643083241) q[8];
rz(-1.6896716822480027) q[8];
ry(3.051514066567441) q[9];
rz(1.8409294588491456) q[9];
ry(0.7512360264917) q[10];
rz(-2.6438572165963232) q[10];
ry(-0.10121919306462068) q[11];
rz(2.5845428240748234) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.5615181725234795) q[0];
rz(2.570001222140675) q[0];
ry(-1.5779133715235476) q[1];
rz(-1.228213380067416) q[1];
ry(-0.7329725506122565) q[2];
rz(-1.3174662361281166) q[2];
ry(1.5732031707886183) q[3];
rz(2.7428546525096356) q[3];
ry(3.1414894882045097) q[4];
rz(-0.4743640742992853) q[4];
ry(1.5666029782518933) q[5];
rz(1.8656295503003069) q[5];
ry(3.1413577144086657) q[6];
rz(0.5875567770373769) q[6];
ry(-3.1415527259639013) q[7];
rz(-0.38635760850549306) q[7];
ry(-1.5738393597000595) q[8];
rz(0.3354349422564731) q[8];
ry(0.003351401361047778) q[9];
rz(-1.992609230178866) q[9];
ry(1.5699009878667827) q[10];
rz(-0.10489566381738145) q[10];
ry(-1.3669161860134977) q[11];
rz(3.032621008862207) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.3044833549430983) q[0];
rz(-1.5025104497533137) q[0];
ry(1.5679988194644823) q[1];
rz(0.3725766537272515) q[1];
ry(-0.000416346567903961) q[2];
rz(1.0088354754300495) q[2];
ry(0.050703969821727306) q[3];
rz(0.39261814580920723) q[3];
ry(3.1392162509528876) q[4];
rz(2.356422820033993) q[4];
ry(-0.000478220562116117) q[5];
rz(-2.007126016135976) q[5];
ry(-3.130804634675074) q[6];
rz(-3.139383584035424) q[6];
ry(-1.5667464043238137) q[7];
rz(2.5157260480720653) q[7];
ry(1.5832646519196478) q[8];
rz(-1.7221026639604922) q[8];
ry(-0.028404424780376306) q[9];
rz(-0.04228835408442677) q[9];
ry(1.5983622038309488) q[10];
rz(0.7980915013066285) q[10];
ry(-1.6011086388822395) q[11];
rz(1.5720532870383224) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5379329853445638) q[0];
rz(1.2262460667322044) q[0];
ry(-0.180854309587627) q[1];
rz(-2.005652462772786) q[1];
ry(-3.127489600694223) q[2];
rz(0.46865473442976846) q[2];
ry(3.1411858751765074) q[3];
rz(0.32925729510299667) q[3];
ry(-3.1414630049557855) q[4];
rz(-1.090292410166625) q[4];
ry(2.0893574260279424) q[5];
rz(-2.1616532587914326) q[5];
ry(1.5560791905446338) q[6];
rz(1.9800952648217165) q[6];
ry(-0.0005104498131300871) q[7];
rz(-1.686577299225844) q[7];
ry(3.1411442641294327) q[8];
rz(-1.1952794067633183) q[8];
ry(0.02114559481744127) q[9];
rz(1.6519255775831052) q[9];
ry(-0.0038533859266370713) q[10];
rz(0.48843715282689404) q[10];
ry(1.568895587451883) q[11];
rz(1.574422948605701) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.041804728934344) q[0];
rz(-0.20228518625639058) q[0];
ry(-1.490676050522377) q[1];
rz(-1.3516886431114532) q[1];
ry(-0.000560606821490637) q[2];
rz(1.8303494315069857) q[2];
ry(-3.1381535831675076) q[3];
rz(1.1600312638893993) q[3];
ry(0.00670613836173839) q[4];
rz(-0.3719281447139169) q[4];
ry(0.0002442625146850119) q[5];
rz(0.5205609590342436) q[5];
ry(-0.00019352031211396705) q[6];
rz(1.1617024023890365) q[6];
ry(0.0002531114078285768) q[7];
rz(-0.7023752143103827) q[7];
ry(0.00036910322722857034) q[8];
rz(3.1027172082808385) q[8];
ry(-2.93117766669725) q[9];
rz(1.6156093045639874) q[9];
ry(1.6046812629517024) q[10];
rz(1.6852866474163486) q[10];
ry(-1.5711793736103454) q[11];
rz(1.5565453040730919) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5103957780746666) q[0];
rz(-0.8218872524188753) q[0];
ry(-0.31616818033017147) q[1];
rz(-2.3443517756929912) q[1];
ry(3.116408741614362) q[2];
rz(2.6086124812845286) q[2];
ry(1.562978978359535) q[3];
rz(-1.5672500545532033) q[3];
ry(-3.1411020574778368) q[4];
rz(-1.2558328931634088) q[4];
ry(-1.44815733685085) q[5];
rz(-1.9793527513289497) q[5];
ry(-1.5550026582386123) q[6];
rz(-1.6196554514599084) q[6];
ry(0.0007285524490363571) q[7];
rz(3.0130011057751562) q[7];
ry(-0.0009167362822428917) q[8];
rz(2.6294942651996616) q[8];
ry(0.003518099510525067) q[9];
rz(3.101820452853581) q[9];
ry(1.5714268312702482) q[10];
rz(-1.6025201570571248) q[10];
ry(-3.1408747343936327) q[11];
rz(-3.0318015962396583) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.1413158101736007) q[0];
rz(-0.7136914741903954) q[0];
ry(-0.004628378554989432) q[1];
rz(0.3490897846361908) q[1];
ry(-0.18921566866003747) q[2];
rz(-2.9884248742361312) q[2];
ry(-0.46954306498885606) q[3];
rz(7.978499208035567e-05) q[3];
ry(-0.17935270609642753) q[4];
rz(-1.5465023705037062) q[4];
ry(-0.0006108370924706046) q[5];
rz(-0.5954669844072031) q[5];
ry(0.0012821768146924482) q[6];
rz(0.3434734353856964) q[6];
ry(-1.5759221627176503) q[7];
rz(-3.139374469046575) q[7];
ry(1.547999326286467) q[8];
rz(-0.7507965188857803) q[8];
ry(-1.5368046047779544) q[9];
rz(1.5857450619410844) q[9];
ry(1.466618156294377) q[10];
rz(-1.0684845530003741) q[10];
ry(-1.5660553597988356) q[11];
rz(1.5989247479571267) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1409787297874145) q[0];
rz(-1.7418115242141083) q[0];
ry(-0.00032024587048429254) q[1];
rz(-2.5193138817734884) q[1];
ry(3.066795488080803) q[2];
rz(-1.2708497398227276) q[2];
ry(1.578932363534347) q[3];
rz(-3.134974523052673) q[3];
ry(0.05449931350927972) q[4];
rz(-2.443164535338259) q[4];
ry(1.5748144485841542) q[5];
rz(1.3055474147972648) q[5];
ry(-0.0022864518373680717) q[6];
rz(2.6888961701397127) q[6];
ry(1.572554644177952) q[7];
rz(1.0761362669606773) q[7];
ry(-3.1413946787568214) q[8];
rz(1.1536215452632173) q[8];
ry(1.5684235970594909) q[9];
rz(2.5452304844832754) q[9];
ry(-0.0008601259681517702) q[10];
rz(-2.2192826164810198) q[10];
ry(1.5689750792105535) q[11];
rz(1.571991512888921) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5710126992314546) q[0];
rz(-2.010447177776718) q[0];
ry(-1.5716885087804164) q[1];
rz(1.3853496503874194) q[1];
ry(-3.1412766571260127) q[2];
rz(0.14466753505551733) q[2];
ry(-1.56863213481311) q[3];
rz(0.06389415805565955) q[3];
ry(-3.1415359275812205) q[4];
rz(2.2516027842708928) q[4];
ry(-3.1415889244273654) q[5];
rz(1.3010983307278763) q[5];
ry(3.126375251084447) q[6];
rz(-2.3817820124843325) q[6];
ry(-3.141524035449118) q[7];
rz(-2.064925600773073) q[7];
ry(2.9566957875781683) q[8];
rz(1.6034028291757116) q[8];
ry(-3.141401002908098) q[9];
rz(2.5453868809202853) q[9];
ry(3.1411999473666916) q[10];
rz(2.4071909972645793) q[10];
ry(0.00010994806556197388) q[11];
rz(2.1797177309760984) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.139395928806409) q[0];
rz(2.6980138521438195) q[0];
ry(0.0022207062579182586) q[1];
rz(-2.955309786653012) q[1];
ry(1.5707695809437814) q[2];
rz(2.5194942278320873) q[2];
ry(3.1157608451826375) q[3];
rz(-3.073769429167743) q[3];
ry(0.023895967782939075) q[4];
rz(-1.3452022544331497) q[4];
ry(0.468142464086635) q[5];
rz(0.8607191016207327) q[5];
ry(0.0006299288599701195) q[6];
rz(1.244409467671466) q[6];
ry(-1.5718694922676262) q[7];
rz(0.0041787092056635436) q[7];
ry(3.1415481123392968) q[8];
rz(2.755827349954414) q[8];
ry(1.5730999494014417) q[9];
rz(3.1262233943022437) q[9];
ry(-0.006197423673248494) q[10];
rz(-1.281442113655201) q[10];
ry(-3.140832243802939) q[11];
rz(0.3243327711897042) q[11];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.57817846238982) q[0];
rz(0.6748036810266753) q[0];
ry(-1.5666220851376078) q[1];
rz(0.6815817205971486) q[1];
ry(-0.006194908804933341) q[2];
rz(-1.8458853044597952) q[2];
ry(-1.571888628732446) q[3];
rz(-0.8896630689660664) q[3];
ry(-0.011502297583367849) q[4];
rz(0.4840920141883069) q[4];
ry(-0.004092417012141058) q[5];
rz(-1.7420765496475912) q[5];
ry(-3.1377023935632415) q[6];
rz(1.2587895532162916) q[6];
ry(-1.573043845372469) q[7];
rz(-0.8882818127148252) q[7];
ry(0.008386317839648605) q[8];
rz(-2.051419367553204) q[8];
ry(-1.5681062048904562) q[9];
rz(-2.458800368940149) q[9];
ry(-1.5694457117531284) q[10];
rz(-0.9006005640732178) q[10];
ry(0.016498470387246833) q[11];
rz(2.536391664862889) q[11];