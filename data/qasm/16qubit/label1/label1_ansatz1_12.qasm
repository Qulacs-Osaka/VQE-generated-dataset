OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.853829349863468) q[0];
rz(1.4532886629846091) q[0];
ry(0.1295635731882543) q[1];
rz(-2.4954437229088042) q[1];
ry(1.3402279908920969) q[2];
rz(0.1282639721922402) q[2];
ry(3.1323623305286907) q[3];
rz(-0.9140873748822191) q[3];
ry(2.746782579180152) q[4];
rz(2.2079953071776615) q[4];
ry(0.011756260122350424) q[5];
rz(2.288577436659034) q[5];
ry(-1.3754976733748139) q[6];
rz(-1.7222738229131647) q[6];
ry(-0.805961367411194) q[7];
rz(2.5268943982732464) q[7];
ry(-0.7279457219040868) q[8];
rz(-1.5749711466325431) q[8];
ry(2.640030893749702) q[9];
rz(0.7716056347837172) q[9];
ry(-0.16551165425404302) q[10];
rz(2.0511857219250373) q[10];
ry(0.08448570831485473) q[11];
rz(0.09075301829499605) q[11];
ry(3.1009846031886688) q[12];
rz(2.863502382624672) q[12];
ry(0.3803285641505726) q[13];
rz(-0.06422281141552982) q[13];
ry(1.0929222310715287) q[14];
rz(0.8362726312379055) q[14];
ry(3.078445503703148) q[15];
rz(-2.839288771482267) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.466165560812305) q[0];
rz(1.7539853001214736) q[0];
ry(0.18698441649723702) q[1];
rz(1.2026029397796139) q[1];
ry(1.1863032385322931) q[2];
rz(2.449981241469912) q[2];
ry(3.1308989511593612) q[3];
rz(-0.6701730795645752) q[3];
ry(1.6343674418615386) q[4];
rz(3.1208216750005278) q[4];
ry(-1.7820742379413332) q[5];
rz(-2.2841313175079834) q[5];
ry(2.235384340587217) q[6];
rz(-1.0089703666001313) q[6];
ry(2.5270751857559186) q[7];
rz(-0.4453231846561865) q[7];
ry(-1.2870180594256846) q[8];
rz(-1.3215947755538193) q[8];
ry(-1.9955019606862359) q[9];
rz(-2.535465857525301) q[9];
ry(1.7003922688740198) q[10];
rz(-1.6983978776122362) q[10];
ry(0.0880385116809208) q[11];
rz(0.18083146251642826) q[11];
ry(0.13178901297863588) q[12];
rz(-0.1910398784183219) q[12];
ry(2.735847003397463) q[13];
rz(0.16045830616458545) q[13];
ry(-1.2997623499269686) q[14];
rz(-2.619084411357632) q[14];
ry(-0.24415079370878132) q[15];
rz(1.0654201254504572) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.755080448199526) q[0];
rz(-2.4383931492568074) q[0];
ry(-0.6787982803581327) q[1];
rz(1.0218581071151123) q[1];
ry(-0.10831160875925681) q[2];
rz(-0.12159646783021409) q[2];
ry(-0.717119930523502) q[3];
rz(1.5932770856040763) q[3];
ry(-0.05267245622483685) q[4];
rz(1.6924932928935974) q[4];
ry(-0.011060397540410884) q[5];
rz(0.339243655170784) q[5];
ry(-3.0561814045236493) q[6];
rz(0.38812671947717714) q[6];
ry(-2.2679585389363286) q[7];
rz(-2.0617441091446436) q[7];
ry(-1.2639663458891786) q[8];
rz(-3.054207227566726) q[8];
ry(0.0749100378524563) q[9];
rz(-1.5633401221432193) q[9];
ry(1.7979758280017566) q[10];
rz(0.57014912184239) q[10];
ry(2.4230456005839716) q[11];
rz(-1.2282926767575733) q[11];
ry(3.0356731101906895) q[12];
rz(0.397914073340635) q[12];
ry(-3.1296947331650515) q[13];
rz(-1.742356075720717) q[13];
ry(-2.9499819701117533) q[14];
rz(-2.9778917213278726) q[14];
ry(2.962741281324559) q[15];
rz(2.0642291451309056) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1056233683269143) q[0];
rz(-0.2937015643605374) q[0];
ry(1.2848505739968372) q[1];
rz(-1.1765103663103087) q[1];
ry(2.933631765308956) q[2];
rz(2.1026471027369933) q[2];
ry(-0.2235182701329288) q[3];
rz(-3.031023172686994) q[3];
ry(-0.17367340204114345) q[4];
rz(0.18205389721861492) q[4];
ry(0.4232676049105164) q[5];
rz(-2.406000426340843) q[5];
ry(0.7492169886301533) q[6];
rz(-0.08942507297385191) q[6];
ry(-1.8175675595116279) q[7];
rz(1.8136539415122181) q[7];
ry(1.4831403909438754) q[8];
rz(-2.237834960665081) q[8];
ry(-2.1784848382696675) q[9];
rz(1.5624241160879135) q[9];
ry(0.06910575466481116) q[10];
rz(2.1726303969651335) q[10];
ry(-3.1258055056247294) q[11];
rz(-2.3708531984637986) q[11];
ry(2.654097758168537) q[12];
rz(2.143186231797489) q[12];
ry(1.732511237172646) q[13];
rz(2.505113024886591) q[13];
ry(0.9918911528072807) q[14];
rz(-1.1497978113577705) q[14];
ry(-1.4273586285635664) q[15];
rz(1.693173649564164) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.6873513777987714) q[0];
rz(-1.745125661072481) q[0];
ry(-1.881256533589033) q[1];
rz(-2.539432162111717) q[1];
ry(0.04015092261956734) q[2];
rz(-1.5706676073552657) q[2];
ry(1.2010545722752526) q[3];
rz(-1.6536373941163434) q[3];
ry(3.0568253801622385) q[4];
rz(0.8549055798021065) q[4];
ry(-0.008853029090924535) q[5];
rz(-1.95465342435806) q[5];
ry(1.6312828991243695) q[6];
rz(-0.022551666264111838) q[6];
ry(0.25075844951792625) q[7];
rz(-1.5967994083160626) q[7];
ry(1.0853495586955266) q[8];
rz(-0.023904820271763445) q[8];
ry(3.0699445270631953) q[9];
rz(-1.3960609557929864) q[9];
ry(-0.7812201394024798) q[10];
rz(2.035198936590855) q[10];
ry(0.6694387121734389) q[11];
rz(-2.13365539316485) q[11];
ry(3.1116898382446507) q[12];
rz(-2.9606420434998664) q[12];
ry(1.5189955559556514) q[13];
rz(-1.2839260196630256) q[13];
ry(3.068003855378343) q[14];
rz(1.2128485108657483) q[14];
ry(-0.2788980497849005) q[15];
rz(-1.5234162636783957) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.1561373014705583) q[0];
rz(-1.2316418028842293) q[0];
ry(0.043305336884167836) q[1];
rz(2.7543979748973753) q[1];
ry(1.024986519887265) q[2];
rz(-2.6990425781087706) q[2];
ry(0.4574437124604387) q[3];
rz(-1.040310566832115) q[3];
ry(2.574718532933964) q[4];
rz(-1.8951054918284456) q[4];
ry(-0.7897996966367504) q[5];
rz(1.9477414514904599) q[5];
ry(1.2497959007371415) q[6];
rz(0.01802889881259695) q[6];
ry(1.4434546052448614) q[7];
rz(-0.11676161816975748) q[7];
ry(1.2327536632522857) q[8];
rz(-0.3626609286047832) q[8];
ry(-2.267300865349129) q[9];
rz(-3.0130548624746782) q[9];
ry(3.1279846442793655) q[10];
rz(0.05854957804404748) q[10];
ry(-3.120644641087231) q[11];
rz(-2.9131771781647324) q[11];
ry(-2.921693513190238) q[12];
rz(-0.20585108283210207) q[12];
ry(-2.741893364469978) q[13];
rz(0.1883954951913331) q[13];
ry(1.7188922765064492) q[14];
rz(-3.020810432139238) q[14];
ry(-1.2695258167498906) q[15];
rz(-3.048249601829101) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.573456376912817) q[0];
rz(-0.6420295436236216) q[0];
ry(-2.0514904068167557) q[1];
rz(-2.60794492100753) q[1];
ry(-3.136635882423445) q[2];
rz(1.0602856653233785) q[2];
ry(-3.114530392227759) q[3];
rz(-0.5526086218819277) q[3];
ry(-0.006126776383752919) q[4];
rz(1.1639703519488265) q[4];
ry(-2.978266039330663) q[5];
rz(-3.0222396960747338) q[5];
ry(1.518579733154172) q[6];
rz(-2.3153683589996037) q[6];
ry(-3.1239339798298715) q[7];
rz(-0.10993889108196697) q[7];
ry(-1.7259065547255776) q[8];
rz(-2.3144959884715135) q[8];
ry(1.1275567796104262) q[9];
rz(0.08898495708323226) q[9];
ry(1.2159156227819214) q[10];
rz(2.4866698244787724) q[10];
ry(-0.23035183789408273) q[11];
rz(0.06498904610148237) q[11];
ry(-3.1050252947535806) q[12];
rz(-2.65056396891355) q[12];
ry(1.773659965295285) q[13];
rz(-1.6047125389638985) q[13];
ry(-2.0053928572895963) q[14];
rz(2.276479884049812) q[14];
ry(2.621872173794693) q[15];
rz(0.09471029821434929) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.0850077652254413) q[0];
rz(0.3500102870075666) q[0];
ry(2.8107691995737705) q[1];
rz(-0.43437629621091417) q[1];
ry(-1.314720027559713) q[2];
rz(-2.8581525052838965) q[2];
ry(-2.8692915273492114) q[3];
rz(-1.3376134854174568) q[3];
ry(-1.728958254290565) q[4];
rz(0.5992435707535595) q[4];
ry(-1.5573732455528833) q[5];
rz(-1.936950230034829) q[5];
ry(-0.003776628927821889) q[6];
rz(-2.3997612201993475) q[6];
ry(2.208014848430004) q[7];
rz(-3.113500419521254) q[7];
ry(-3.1383738783201194) q[8];
rz(-0.7512109211088269) q[8];
ry(-1.5820579962917103) q[9];
rz(1.5818629872410213) q[9];
ry(-2.1597102899928444) q[10];
rz(-0.39649889431895585) q[10];
ry(-2.209801542925664) q[11];
rz(-1.1968032621312072) q[11];
ry(-0.9593961185668727) q[12];
rz(1.8884899509039195) q[12];
ry(1.464121716739079) q[13];
rz(-1.7205928630449172) q[13];
ry(0.25994732550999355) q[14];
rz(-3.033708023358592) q[14];
ry(0.2732773894684613) q[15];
rz(2.2434895901127625) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.8303833828985608) q[0];
rz(0.9311076197205805) q[0];
ry(-1.249434313036007) q[1];
rz(-3.1082132505712115) q[1];
ry(-3.10959019199527) q[2];
rz(1.9873282224458004) q[2];
ry(-3.1137250777098395) q[3];
rz(-1.009015146987409) q[3];
ry(1.0328413895590787) q[4];
rz(2.6943309501215467) q[4];
ry(2.9500042194711322) q[5];
rz(1.195745160226074) q[5];
ry(0.20440924868089755) q[6];
rz(1.6384447184640978) q[6];
ry(-1.6519256524333699) q[7];
rz(1.565701907671454) q[7];
ry(-1.4640980393905374) q[8];
rz(-3.056998315990383) q[8];
ry(-1.4949031190474695) q[9];
rz(2.009307753903052) q[9];
ry(-1.593473774324524) q[10];
rz(-1.4221366516740055) q[10];
ry(0.3686317161919028) q[11];
rz(0.2770261334537824) q[11];
ry(0.8233785737238772) q[12];
rz(-1.3347356430275186) q[12];
ry(-0.09428904886414384) q[13];
rz(-3.119248215519744) q[13];
ry(-1.7066568201069912) q[14];
rz(-1.7737516649491036) q[14];
ry(-0.6295559317928463) q[15];
rz(-0.4869198155176848) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.556022806258122) q[0];
rz(1.6358847226090898) q[0];
ry(-2.2089058199353806) q[1];
rz(-2.9140446336448624) q[1];
ry(-0.8391911020070583) q[2];
rz(1.522085398361759) q[2];
ry(2.1105700065250925) q[3];
rz(2.634966717718771) q[3];
ry(-1.9086801160085667) q[4];
rz(-2.17775383911729) q[4];
ry(-1.5761229260772778) q[5];
rz(-3.1185767908784827) q[5];
ry(-1.6808956216807984) q[6];
rz(-0.09434863346184304) q[6];
ry(-2.898485196896623) q[7];
rz(-1.4739434121523372) q[7];
ry(-2.9905164498152974) q[8];
rz(2.8099928382356194) q[8];
ry(3.0834098257071036) q[9];
rz(2.006800468363089) q[9];
ry(0.02885817252862746) q[10];
rz(-1.7136580038762128) q[10];
ry(1.590654696837479) q[11];
rz(3.1218043667778277) q[11];
ry(-1.9009569229796046) q[12];
rz(0.16358132562054972) q[12];
ry(-2.8952025207176764) q[13];
rz(2.940844263636732) q[13];
ry(1.508003189979549) q[14];
rz(-2.063429191275212) q[14];
ry(3.027231817903609) q[15];
rz(1.7763869074111467) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.923918085504014) q[0];
rz(-1.418134845387885) q[0];
ry(2.7878653831277127) q[1];
rz(-0.4843055722500491) q[1];
ry(-0.7224668135706303) q[2];
rz(0.9405764817313669) q[2];
ry(3.0816725011475046) q[3];
rz(-1.8090187433307703) q[3];
ry(1.563126098584445) q[4];
rz(0.03463385163754573) q[4];
ry(2.3101137865500996) q[5];
rz(0.5196098685497337) q[5];
ry(-3.087978513832665) q[6];
rz(2.35430848185358) q[6];
ry(3.0760130428104584) q[7];
rz(0.039212599549687255) q[7];
ry(-1.6932679489106914) q[8];
rz(-1.54544469112792) q[8];
ry(-1.5026972153661438) q[9];
rz(1.5743591156907537) q[9];
ry(-2.753123318020509) q[10];
rz(1.5963041233701478) q[10];
ry(3.0932158910497933) q[11];
rz(-1.5320783854001063) q[11];
ry(1.5683700835391008) q[12];
rz(3.141032059885246) q[12];
ry(1.9323002353792469) q[13];
rz(0.575494514547267) q[13];
ry(0.8974661254031435) q[14];
rz(-0.1249056731730689) q[14];
ry(2.9697794893900515) q[15];
rz(-0.25294300486142246) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.9246026791197075) q[0];
rz(1.6997255031544798) q[0];
ry(2.2418558537943136) q[1];
rz(-3.0235649362742296) q[1];
ry(1.8748261024588078) q[2];
rz(3.083678129620515) q[2];
ry(-1.569393169777693) q[3];
rz(-3.1325231284640047) q[3];
ry(-0.5017241688693538) q[4];
rz(-0.28316428902099344) q[4];
ry(1.8435379349625607) q[5];
rz(2.8034862296046823) q[5];
ry(-0.06363027465300758) q[6];
rz(-2.662396547780504) q[6];
ry(-1.8682910112067788) q[7];
rz(-0.008171582279809382) q[7];
ry(-2.229358182918029) q[8];
rz(3.135290993370093) q[8];
ry(1.397406085011917) q[9];
rz(1.4008904468672707) q[9];
ry(-1.5707835489458748) q[10];
rz(-2.2126704245592506) q[10];
ry(1.8118699952660817) q[11];
rz(3.0841575311579645) q[11];
ry(-3.028924354570389) q[12];
rz(-1.752811386608225) q[12];
ry(1.5799439643041708) q[13];
rz(-1.5883709350279949) q[13];
ry(-0.3513980692755845) q[14];
rz(0.32830958590628884) q[14];
ry(-3.0610230785053973) q[15];
rz(1.7412519673156275) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.49444486985954145) q[0];
rz(2.2388758697812086) q[0];
ry(-0.39662075781429357) q[1];
rz(-2.0674655488517173) q[1];
ry(1.567405190400498) q[2];
rz(-3.1389084654020034) q[2];
ry(-0.10908181210182764) q[3];
rz(-1.5795098856136436) q[3];
ry(0.053124328061698194) q[4];
rz(1.8547019877219582) q[4];
ry(-0.9064250974959319) q[5];
rz(-0.014405675508251474) q[5];
ry(0.03065534827687788) q[6];
rz(1.8352150395743276) q[6];
ry(2.9705671138049943) q[7];
rz(1.616973887950649) q[7];
ry(-0.09165440220966983) q[8];
rz(0.09368537858611514) q[8];
ry(0.02910655711943599) q[9];
rz(0.13549526462450867) q[9];
ry(-3.0136573716913344) q[10];
rz(2.7005048446703857) q[10];
ry(-3.016506021214638) q[11];
rz(-0.35067304113174197) q[11];
ry(3.1315643293968844) q[12];
rz(-0.20773768794309286) q[12];
ry(-1.5788635017291723) q[13];
rz(-0.5847714861644411) q[13];
ry(1.5788474537485433) q[14];
rz(1.5706479723127689) q[14];
ry(0.7766196421098837) q[15];
rz(2.8814300998810585) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.886758831609379) q[0];
rz(-0.6959280187990879) q[0];
ry(-1.5727510189150424) q[1];
rz(-3.1392591647127888) q[1];
ry(2.152111231988547) q[2];
rz(-1.5757287363014907) q[2];
ry(-1.5675234175392883) q[3];
rz(0.1459441875133931) q[3];
ry(3.0020633602042475) q[4];
rz(0.029068528907398628) q[4];
ry(2.270289211070158) q[5];
rz(0.9583271033554811) q[5];
ry(-2.870560662423035) q[6];
rz(0.02015397123130324) q[6];
ry(1.6255942830725014) q[7];
rz(3.014851225079301) q[7];
ry(3.118740626656578) q[8];
rz(2.6414571003763405) q[8];
ry(1.9257766577046267) q[9];
rz(1.4153811928475406) q[9];
ry(-0.012519587691255165) q[10];
rz(-2.8989230304216593) q[10];
ry(1.836154043547605) q[11];
rz(-0.03616820050711933) q[11];
ry(2.7338958826256063) q[12];
rz(-0.02963350752649147) q[12];
ry(-1.5701654562361493) q[13];
rz(1.5859473021074086) q[13];
ry(-2.011320270712872) q[14];
rz(1.8050466677723005) q[14];
ry(0.0011838812584584555) q[15];
rz(1.2229179104555448) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.5726237285268674) q[0];
rz(-1.5686501949768523) q[0];
ry(-2.7060300806041835) q[1];
rz(1.5749801839205713) q[1];
ry(-0.12121635317094381) q[2];
rz(-3.1354821601131504) q[2];
ry(-0.0008099975670522941) q[3];
rz(2.6723354972029236) q[3];
ry(0.1384616173532072) q[4];
rz(0.009628537972941942) q[4];
ry(-2.5970077165416092) q[5];
rz(-0.13720234457595915) q[5];
ry(0.09689870185878408) q[6];
rz(-3.1033777265848568) q[6];
ry(-3.134637418532198) q[7];
rz(-0.6912079560355445) q[7];
ry(-0.00567043695592595) q[8];
rz(2.155545051843164) q[8];
ry(3.1335571892892458) q[9];
rz(-1.6023392825312015) q[9];
ry(-0.09562359084363958) q[10];
rz(2.570327948665917) q[10];
ry(0.18088965926466444) q[11];
rz(0.017552963667013977) q[11];
ry(2.9067866522239165) q[12];
rz(-1.56257327819994) q[12];
ry(3.0050133749373047) q[13];
rz(-1.5480009762969473) q[13];
ry(0.018569526904418196) q[14];
rz(2.905707780833961) q[14];
ry(-0.5117834285824484) q[15];
rz(1.5792982839357708) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5712884266022593) q[0];
rz(0.7871490650964956) q[0];
ry(1.5696273406982633) q[1];
rz(1.856938627083405) q[1];
ry(-1.570997090107196) q[2];
rz(-1.1734390929332388) q[2];
ry(-3.136292619403953) q[3];
rz(1.149070476943411) q[3];
ry(-1.7121709903719982) q[4];
rz(-0.6140202874463361) q[4];
ry(1.5835129305817794) q[5];
rz(-0.7792396240638312) q[5];
ry(-1.3320870381653416) q[6];
rz(-0.5255278577043363) q[6];
ry(0.07117459927696324) q[7];
rz(-0.27389070774118895) q[7];
ry(-1.529253696017066) q[8];
rz(-1.9904441269994138) q[8];
ry(0.346975694355872) q[9];
rz(-2.4108468201570137) q[9];
ry(0.0021025260379348154) q[10];
rz(-1.6292258523577696) q[10];
ry(-1.5525394243323225) q[11];
rz(0.8453887884422037) q[11];
ry(1.5520592194516472) q[12];
rz(2.7883846867788415) q[12];
ry(-1.5797664731255054) q[13];
rz(-1.2706489641208725) q[13];
ry(1.5721881584527664) q[14];
rz(-0.15824593283455246) q[14];
ry(3.141254287662208) q[15];
rz(-2.261136096475599) q[15];