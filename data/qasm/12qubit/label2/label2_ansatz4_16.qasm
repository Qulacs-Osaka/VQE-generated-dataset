OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.492352551966449) q[0];
rz(-0.5956409116840399) q[0];
ry(-2.887465469035271) q[1];
rz(2.469997714601051) q[1];
ry(3.1404785728023854) q[2];
rz(-1.2785044638508327) q[2];
ry(3.1409587936621093) q[3];
rz(-1.1759090887552182) q[3];
ry(-1.5736672293245748) q[4];
rz(-1.7340887061503008) q[4];
ry(-1.579585570009204) q[5];
rz(2.7166888498639445) q[5];
ry(-3.1415485080385093) q[6];
rz(-2.4477549232053977) q[6];
ry(-0.0007630165469345096) q[7];
rz(-2.727616629518815) q[7];
ry(0.007265152421967791) q[8];
rz(1.0893629469092543) q[8];
ry(0.0018931808857987065) q[9];
rz(-0.02181779760934344) q[9];
ry(-1.5645310533861503) q[10];
rz(2.70792924643155) q[10];
ry(1.5661180624830333) q[11];
rz(-0.42028639606067586) q[11];
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
ry(-0.7930684053675652) q[0];
rz(0.015497363656262968) q[0];
ry(-1.7034441940846825) q[1];
rz(0.6380267655891637) q[1];
ry(-0.0039828645266668265) q[2];
rz(1.1391193294853155) q[2];
ry(0.01344469069577059) q[3];
rz(-1.686090155878687) q[3];
ry(-1.8107609331620322) q[4];
rz(1.7727637199011397) q[4];
ry(2.2194614637167613) q[5];
rz(-3.044720245975799) q[5];
ry(-3.0121024069072924) q[6];
rz(-2.981724180343959) q[6];
ry(-0.6076650412951483) q[7];
rz(1.0733631681490232) q[7];
ry(1.5884013142711417) q[8];
rz(-0.16129530976701145) q[8];
ry(-1.5836957273755443) q[9];
rz(-1.3111185546765194) q[9];
ry(2.113516582008427) q[10];
rz(-3.0724841066919963) q[10];
ry(-0.9022645246740201) q[11];
rz(-0.35432054208617286) q[11];
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
ry(-1.7987571239028757) q[0];
rz(-1.53179804879628) q[0];
ry(-2.664991598267732) q[1];
rz(0.22138935390000913) q[1];
ry(0.0020715145116438904) q[2];
rz(2.635101060423419) q[2];
ry(3.13449405113208) q[3];
rz(-0.9205465700480023) q[3];
ry(3.1396254054441513) q[4];
rz(1.3648868573693822) q[4];
ry(-0.010321092761633921) q[5];
rz(-1.672961845198099) q[5];
ry(-0.0008753825309355451) q[6];
rz(-0.8633523392997965) q[6];
ry(3.1409294320715264) q[7];
rz(-2.9296393242728813) q[7];
ry(3.1306775058305853) q[8];
rz(1.4449811206453373) q[8];
ry(-0.03222802396847102) q[9];
rz(2.7838139306961027) q[9];
ry(-0.5233622908051163) q[10];
rz(-0.1689313250799618) q[10];
ry(-2.9970110639745413) q[11];
rz(1.12350230066503) q[11];
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
ry(-2.8530245529631992) q[0];
rz(-1.8245188124128364) q[0];
ry(2.8790127185970333) q[1];
rz(2.2842190277513423) q[1];
ry(-1.4552949330106708) q[2];
rz(0.4292530863272099) q[2];
ry(1.280190160640851) q[3];
rz(1.4757127456160495) q[3];
ry(0.14834272636792323) q[4];
rz(-2.2449341004422214) q[4];
ry(-1.928904410448065) q[5];
rz(2.0209625300833007) q[5];
ry(-0.15258549414712738) q[6];
rz(-2.5950327956419263) q[6];
ry(-0.3133615042123603) q[7];
rz(2.9202380211168357) q[7];
ry(-3.078572360487829) q[8];
rz(1.6160572444407626) q[8];
ry(0.018075262625230515) q[9];
rz(1.5811077363049) q[9];
ry(-0.3313838836289033) q[10];
rz(1.7161257707791504) q[10];
ry(-0.025845953330879076) q[11];
rz(-0.42355640331791333) q[11];
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
ry(2.4778994334916207) q[0];
rz(-2.3552192282658475) q[0];
ry(2.5255909334141124) q[1];
rz(-2.410070709480166) q[1];
ry(2.0172302509579905) q[2];
rz(-0.5690430785134575) q[2];
ry(-1.8450592571126947) q[3];
rz(-1.8361085907347545) q[3];
ry(3.1295225447822506) q[4];
rz(-2.82842472242608) q[4];
ry(3.128761595549432) q[5];
rz(-1.3274116759767878) q[5];
ry(1.5693973667580339) q[6];
rz(-1.8378691087153747) q[6];
ry(1.5606143620824355) q[7];
rz(-2.6333225148602732) q[7];
ry(-1.140729736225482) q[8];
rz(-1.6908261425520283) q[8];
ry(1.9917910073222251) q[9];
rz(-0.43115866548944837) q[9];
ry(-1.7854558179907654) q[10];
rz(-3.1077304635697516) q[10];
ry(1.856473247820591) q[11];
rz(-2.566382841252666) q[11];
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
ry(1.3487028304865563) q[0];
rz(-2.4215737884844883) q[0];
ry(-1.8424806528019593) q[1];
rz(2.5580533911764247) q[1];
ry(2.731583725747743) q[2];
rz(0.7860633852580621) q[2];
ry(2.8960703401027112) q[3];
rz(1.1318140058538269) q[3];
ry(3.0617698729402965) q[4];
rz(-0.3970757473647729) q[4];
ry(-0.034931950023247005) q[5];
rz(2.1675168268516534) q[5];
ry(2.62259464980729) q[6];
rz(1.2136169279870028) q[6];
ry(3.0036909558760767) q[7];
rz(-0.878261892235372) q[7];
ry(1.571926926406879) q[8];
rz(-2.32295522088934) q[8];
ry(0.010322671063460296) q[9];
rz(-2.1571393256897773) q[9];
ry(2.1968343482043124) q[10];
rz(-2.5286019772142496) q[10];
ry(2.1226663386814097) q[11];
rz(1.9034372664171233) q[11];
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
ry(1.0062289308309804) q[0];
rz(0.6311608702378572) q[0];
ry(-0.9775967880575047) q[1];
rz(0.9923462445116514) q[1];
ry(-1.6538038073301058) q[2];
rz(2.6955323618091676) q[2];
ry(-0.06786020138454447) q[3];
rz(1.640350099122974) q[3];
ry(-0.004089974132989414) q[4];
rz(-2.4049020459995734) q[4];
ry(-3.133845716171641) q[5];
rz(0.8308901569059115) q[5];
ry(0.7769131519037803) q[6];
rz(-1.1878603067110625) q[6];
ry(1.696032762181325) q[7];
rz(-0.17433758185523457) q[7];
ry(-3.0713906810175002) q[8];
rz(0.43698665717295254) q[8];
ry(-0.8172668887623793) q[9];
rz(2.7840518091933535) q[9];
ry(-2.9803696713813386) q[10];
rz(-2.5032340880604766) q[10];
ry(0.1361457887551773) q[11];
rz(-1.962541088051796) q[11];
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
ry(-2.006875470082118) q[0];
rz(-2.2628811823876998) q[0];
ry(-1.8384573741329167) q[1];
rz(0.613046296107158) q[1];
ry(-0.37457141939959937) q[2];
rz(2.5847795892660286) q[2];
ry(2.759726839210717) q[3];
rz(0.02725360157477308) q[3];
ry(1.7172804158326307) q[4];
rz(-0.45731470828634696) q[4];
ry(-1.4742699642305699) q[5];
rz(1.1708608212631413) q[5];
ry(3.137214484383077) q[6];
rz(-2.8141569692911856) q[6];
ry(3.1120657847474917) q[7];
rz(0.42251748212712315) q[7];
ry(-0.014178938761619799) q[8];
rz(-0.0312615919342889) q[8];
ry(-3.131296235873093) q[9];
rz(-2.906802910051917) q[9];
ry(2.9543554330687205) q[10];
rz(-0.31227588758664737) q[10];
ry(-2.9597902286430227) q[11];
rz(0.018146505506718032) q[11];
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
ry(-0.13209547843343272) q[0];
rz(-0.6752189803380362) q[0];
ry(2.628221883717766) q[1];
rz(2.310399346587947) q[1];
ry(0.015978393964776853) q[2];
rz(0.6764923823755788) q[2];
ry(-0.02020598969421794) q[3];
rz(-2.532871558917886) q[3];
ry(-0.03237462611869546) q[4];
rz(-1.948473273828952) q[4];
ry(3.132345868107541) q[5];
rz(-2.8025898640365425) q[5];
ry(-3.110397273679035) q[6];
rz(2.8253969990233423) q[6];
ry(3.12619287226655) q[7];
rz(0.4046269127544838) q[7];
ry(0.8612164754135048) q[8];
rz(-1.3977819246879195) q[8];
ry(1.597161748160989) q[9];
rz(-1.5270259763300524) q[9];
ry(-0.6729765563175443) q[10];
rz(-1.0465312185531612) q[10];
ry(-2.1222162289749673) q[11];
rz(3.1395317087694767) q[11];
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
ry(2.1726037965535205) q[0];
rz(-1.7375880179894105) q[0];
ry(1.6474792574867294) q[1];
rz(1.040382842998533) q[1];
ry(0.5083902935994355) q[2];
rz(2.855058676835552) q[2];
ry(-1.3301406213767164) q[3];
rz(-2.0858997108400703) q[3];
ry(2.020073690196503) q[4];
rz(2.993252652340296) q[4];
ry(1.1179109603939814) q[5];
rz(-1.0506168759093777) q[5];
ry(0.004680816962962029) q[6];
rz(-1.0793539039851714) q[6];
ry(-3.124079831480379) q[7];
rz(1.2616807921614543) q[7];
ry(-3.128680590961245) q[8];
rz(-1.0037009859068686) q[8];
ry(-3.133544760061514) q[9];
rz(-1.1768204657537762) q[9];
ry(3.0991494763259064) q[10];
rz(1.703733215438609) q[10];
ry(0.045898078845325686) q[11];
rz(3.108254393848189) q[11];
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
ry(1.6631222106028898) q[0];
rz(2.361216279677672) q[0];
ry(3.1263218442847758) q[1];
rz(3.1371756001263313) q[1];
ry(3.1389165224619306) q[2];
rz(2.952661861062023) q[2];
ry(-3.139360916835307) q[3];
rz(1.1440361669054437) q[3];
ry(3.0946781235351466) q[4];
rz(1.1056345247158283) q[4];
ry(-0.07912953605698382) q[5];
rz(-1.6680665406238344) q[5];
ry(1.9925616325907127) q[6];
rz(-0.4427244284213261) q[6];
ry(-2.480757645153618) q[7];
rz(0.5774504090289616) q[7];
ry(-2.6154764785487994) q[8];
rz(-0.04371496817173633) q[8];
ry(-1.0257728748136394) q[9];
rz(1.5295885084622851) q[9];
ry(-0.9070815862973858) q[10];
rz(-1.6777789519035666) q[10];
ry(-1.973190975014127) q[11];
rz(-0.16208308238871683) q[11];
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
ry(2.386871299353717) q[0];
rz(-1.2011973807771001) q[0];
ry(-2.0244031416048083) q[1];
rz(1.6296000477728092) q[1];
ry(-3.0214175146585385) q[2];
rz(0.4592261897858201) q[2];
ry(-2.6815715634906456) q[3];
rz(1.4551649051864801) q[3];
ry(-0.28536467373290186) q[4];
rz(-1.5166616751156448) q[4];
ry(3.1207796446848293) q[5];
rz(0.07818331573239372) q[5];
ry(-3.1150224494994614) q[6];
rz(-0.8638672532550546) q[6];
ry(-3.1167188716414413) q[7];
rz(3.052001511577861) q[7];
ry(2.867736876481729) q[8];
rz(-2.520210806260066) q[8];
ry(0.2180000274262479) q[9];
rz(2.2607819003034626) q[9];
ry(-1.6418389246529532) q[10];
rz(0.7605053681525088) q[10];
ry(-0.9160652109143349) q[11];
rz(-1.986656138093931) q[11];
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
ry(1.1884150349935125) q[0];
rz(-2.2979105713120105) q[0];
ry(1.6953896410248648) q[1];
rz(-2.2088758404485542) q[1];
ry(0.025854141543065094) q[2];
rz(-2.59698183256798) q[2];
ry(3.105557522347217) q[3];
rz(1.2759336155544496) q[3];
ry(1.5286351343475355) q[4];
rz(0.7061754467286772) q[4];
ry(-1.586808909522432) q[5];
rz(-0.8295561413128825) q[5];
ry(-0.0073962395449767016) q[6];
rz(-0.21691613132758933) q[6];
ry(-0.22610505620859822) q[7];
rz(1.383816297845479) q[7];
ry(-3.105212154319171) q[8];
rz(-1.0400643815133044) q[8];
ry(0.049876221469741) q[9];
rz(-0.34172873486228905) q[9];
ry(2.90336357413483) q[10];
rz(0.0073480324736255355) q[10];
ry(-0.027024726331414478) q[11];
rz(-2.17095446439305) q[11];
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
ry(0.7362229747428319) q[0];
rz(-0.395308165252489) q[0];
ry(-0.8257391235420881) q[1];
rz(-0.7943504575797729) q[1];
ry(-1.5917554050728417) q[2];
rz(1.4882408262903155) q[2];
ry(-1.4753139638904749) q[3];
rz(-2.731822673841014) q[3];
ry(0.0023516531298159965) q[4];
rz(0.09702400641683795) q[4];
ry(5.928838257846132e-05) q[5];
rz(-1.9560175213485491) q[5];
ry(1.221553873570997) q[6];
rz(0.5939052660232196) q[6];
ry(1.9465181365051327) q[7];
rz(-1.1099010540046412) q[7];
ry(-0.08800746088818467) q[8];
rz(0.6574520397613653) q[8];
ry(-3.032700298556722) q[9];
rz(2.7795819786795124) q[9];
ry(0.853828631871675) q[10];
rz(-0.0637759291407276) q[10];
ry(-0.2445119289141865) q[11];
rz(-0.1561395709395286) q[11];
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
ry(-3.1090219453741317) q[0];
rz(-2.3499010039201917) q[0];
ry(3.1369372399906252) q[1];
rz(0.5687315791900543) q[1];
ry(-0.01095885878694336) q[2];
rz(0.2462597108954672) q[2];
ry(-0.006569727769439203) q[3];
rz(2.860021844875761) q[3];
ry(3.1328452296705667) q[4];
rz(0.794647026022315) q[4];
ry(-0.008329028068064882) q[5];
rz(2.7279210245088628) q[5];
ry(3.056080040085179) q[6];
rz(-2.5166087932315357) q[6];
ry(-2.983026957406041) q[7];
rz(-1.0740517900044377) q[7];
ry(-3.137228669859341) q[8];
rz(-1.8065799455420857) q[8];
ry(0.008574836054954193) q[9];
rz(1.3778827295261378) q[9];
ry(-3.0564645371087837) q[10];
rz(1.6850136915078824) q[10];
ry(-2.9635099142847956) q[11];
rz(1.329497483480999) q[11];
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
ry(1.4244337437536796) q[0];
rz(-0.2464752076751449) q[0];
ry(-1.6190427552805309) q[1];
rz(-0.2674485576619299) q[1];
ry(-1.8171608666135013) q[2];
rz(0.0674597229441618) q[2];
ry(-3.063809389924863) q[3];
rz(-1.4195488251823907) q[3];
ry(-1.521886898677713) q[4];
rz(-0.18561450352838096) q[4];
ry(-1.583839892071619) q[5];
rz(-3.0057678844343454) q[5];
ry(1.90483194674139) q[6];
rz(0.9318810008239626) q[6];
ry(-1.2003212457320043) q[7];
rz(-2.1846537603546548) q[7];
ry(-1.8575396082817797) q[8];
rz(2.280496943407498) q[8];
ry(-1.6580383680333988) q[9];
rz(1.0855344443932484) q[9];
ry(2.5339838484892447) q[10];
rz(1.4181678297093405) q[10];
ry(0.9814233657850134) q[11];
rz(-1.8569546151655612) q[11];
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
ry(-3.1145681083321635) q[0];
rz(-1.0487625416063264) q[0];
ry(3.130587289924446) q[1];
rz(-0.8717577771919354) q[1];
ry(1.6326499873504994) q[2];
rz(1.6991816157075528) q[2];
ry(1.7029811164827515) q[3];
rz(1.5716706213071172) q[3];
ry(3.1323012145941274) q[4];
rz(-1.0334757954100207) q[4];
ry(-3.136119162816779) q[5];
rz(1.8620494826257037) q[5];
ry(-3.1373127779563847) q[6];
rz(-2.6676009948229162) q[6];
ry(3.141362856363422) q[7];
rz(1.2454141951340656) q[7];
ry(-2.5926851926650016) q[8];
rz(1.1794496257703118) q[8];
ry(2.3150404440714105) q[9];
rz(1.728679551103654) q[9];
ry(-0.7623022266359688) q[10];
rz(1.59552090710771) q[10];
ry(-2.623433251022961) q[11];
rz(-1.4040032792966928) q[11];
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
ry(-1.0089272863257355) q[0];
rz(-2.871847573327341) q[0];
ry(-1.4976398691537325) q[1];
rz(-2.64006921169102) q[1];
ry(-1.5341880030151633) q[2];
rz(-2.7538600021887927) q[2];
ry(-1.773062851025854) q[3];
rz(1.6144748835718261) q[3];
ry(0.17047368922699135) q[4];
rz(0.4708288177238531) q[4];
ry(-2.9696659671102683) q[5];
rz(-1.2649556263195292) q[5];
ry(-0.03572071200670113) q[6];
rz(-1.349038259965221) q[6];
ry(-0.0643347217456946) q[7];
rz(0.8497595367462444) q[7];
ry(3.0955645038173745) q[8];
rz(2.1438276748348004) q[8];
ry(-0.031986088717023665) q[9];
rz(-1.5655100454761444) q[9];
ry(2.982824825535333) q[10];
rz(-0.7813367346202078) q[10];
ry(2.9682871895338327) q[11];
rz(0.7992004453889159) q[11];
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
ry(1.0070022699211654) q[0];
rz(-2.661946221143748) q[0];
ry(1.306509101470538) q[1];
rz(1.0392890838621989) q[1];
ry(0.01756095565086195) q[2];
rz(2.584592011468726) q[2];
ry(-3.1048771241764666) q[3];
rz(-0.7602546126733696) q[3];
ry(0.010358502040619832) q[4];
rz(-1.6375249396252343) q[4];
ry(-0.0021507071405917344) q[5];
rz(2.2104520619634007) q[5];
ry(-3.1371238610460526) q[6];
rz(-0.14052891839847173) q[6];
ry(-3.137800179525836) q[7];
rz(-0.562640279856141) q[7];
ry(1.5015720744146068) q[8];
rz(1.5016310645707291) q[8];
ry(2.2371612489200086) q[9];
rz(1.494179894721971) q[9];
ry(-0.888645194976427) q[10];
rz(-1.075726658094415) q[10];
ry(0.7181801821883651) q[11];
rz(-1.8135443164443248) q[11];
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
ry(0.8797663624404057) q[0];
rz(0.775209756073628) q[0];
ry(2.0028305161120095) q[1];
rz(0.26252578242750185) q[1];
ry(-2.10455006257527) q[2];
rz(-2.311325988421574) q[2];
ry(1.7136467634624202) q[3];
rz(-0.8408319376018768) q[3];
ry(1.1944558565675507) q[4];
rz(1.3356315313045002) q[4];
ry(-1.9141291085964833) q[5];
rz(1.3360943978576332) q[5];
ry(0.8853358839758246) q[6];
rz(-2.8858668348904986) q[6];
ry(-2.2041303316686256) q[7];
rz(-2.906335820631849) q[7];
ry(-2.237722720086398) q[8];
rz(0.9560602728909268) q[8];
ry(0.8906535353944411) q[9];
rz(-2.9229815077646983) q[9];
ry(-2.174044805006487) q[10];
rz(1.055894839470824) q[10];
ry(-0.9824484250533034) q[11];
rz(-2.0862610000911057) q[11];