OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1453614613433012) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.023842195465252125) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07584542934145809) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.0009865813437069234) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.014459425267631122) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05019762470079074) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.8684920377330684) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.8378812190319592) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09259258708708007) q[3];
cx q[2],q[3];
rx(-0.04181648323855175) q[0];
rz(-0.0014528585560862949) q[0];
rx(0.17572203925544144) q[1];
rz(-0.08837308571430307) q[1];
rx(-0.12157405077444024) q[2];
rz(0.007912989890606768) q[2];
rx(0.29025205910584917) q[3];
rz(-0.06611058699623215) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1853786276083334) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06034998828762424) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.026209314375500746) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10583884809417968) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.024730850333516206) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.040437905060193784) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5018845081235227) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6213140049307674) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08535303800569269) q[3];
cx q[2],q[3];
rx(0.01750716956803651) q[0];
rz(-0.031088792015332614) q[0];
rx(0.13924779125862713) q[1];
rz(-0.049754113638641614) q[1];
rx(-0.26166943470230675) q[2];
rz(0.008553783589836227) q[2];
rx(0.12062721164317121) q[3];
rz(0.002494755336246221) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15475923598829705) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.027730565450019635) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.029258449621778097) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.05626466956865679) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03091671145471833) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.006752887943084344) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.27891944573285155) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5144439384177248) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.16979934334003022) q[3];
cx q[2],q[3];
rx(0.002169808342598814) q[0];
rz(-0.04090787788423722) q[0];
rx(0.11113672436855743) q[1];
rz(-0.023456117206175197) q[1];
rx(-0.3839476373402897) q[2];
rz(0.001508895173619677) q[2];
rx(0.03375420029728842) q[3];
rz(0.0010346292480672345) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13257596294693824) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.005400961982966428) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03269184062122022) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.25816301995311847) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09826701489917354) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05191342998170409) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.20951300218405625) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1652293102649721) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1531587656183562) q[3];
cx q[2],q[3];
rx(-0.025259127080149782) q[0];
rz(-0.08940284506988938) q[0];
rx(-0.049043405022968105) q[1];
rz(-0.005950158629431324) q[1];
rx(-0.41054510676512446) q[2];
rz(-0.07691901439570722) q[2];
rx(0.08736641749718005) q[3];
rz(0.060741890746404584) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0889049651682903) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.09725205235672343) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1603321056995781) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3703651846177972) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3772254832340821) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.07496711015010014) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.20542997000077184) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.006689342135519719) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06196249791177287) q[3];
cx q[2],q[3];
rx(-0.032244577260549176) q[0];
rz(-0.11733286754261804) q[0];
rx(-0.11263870633918806) q[1];
rz(-0.12764915625649928) q[1];
rx(-0.36337526406241183) q[2];
rz(-0.172268945313642) q[2];
rx(0.03841091709012723) q[3];
rz(0.07254548924503525) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05922507415560722) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2223991730023198) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.28220561234388647) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6127100829059414) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2552414001450675) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.09240153735499682) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4066063044479638) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.10592901347432006) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02296036687563944) q[3];
cx q[2],q[3];
rx(-0.050575746954855075) q[0];
rz(-0.08350184194285964) q[0];
rx(-0.14935496063535542) q[1];
rz(-0.335089020247753) q[1];
rx(-0.3439824204105768) q[2];
rz(-0.18045532096782974) q[2];
rx(0.09413409522039454) q[3];
rz(0.15500676937785587) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.06883098653753514) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.22985422610422307) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.22466219186353273) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7872516914385111) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0017935893922793554) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08421784551336291) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4543822966485453) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11703167656086697) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05348166420936845) q[3];
cx q[2],q[3];
rx(-0.009954277230127378) q[0];
rz(-0.06942184063842868) q[0];
rx(-0.16075828818415533) q[1];
rz(-0.39533839369828516) q[1];
rx(-0.2654618489355072) q[2];
rz(-0.20353945219675157) q[2];
rx(0.08666097221542136) q[3];
rz(0.11294967235206599) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.004456420178729821) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2179387658501756) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.17041570391212327) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9344482556733482) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2400691999113347) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12511161642885865) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.46937217886175336) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.24821443327295153) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.030383573453908755) q[3];
cx q[2],q[3];
rx(-0.029301176868800727) q[0];
rz(-0.01411189838349922) q[0];
rx(-0.12612413190272334) q[1];
rz(-0.31406882139868797) q[1];
rx(-0.04319282445982674) q[2];
rz(-0.3265451916906708) q[2];
rx(0.0897943208925868) q[3];
rz(0.05090085583519539) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0065077734272509435) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.14673631380067417) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1579912015355064) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.8640926692694844) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.376351965806974) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10382625992429927) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4007201111175074) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3294261039145618) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.040125651206661575) q[3];
cx q[2],q[3];
rx(0.016643852282239587) q[0];
rz(0.042067106204792114) q[0];
rx(0.05997217509291085) q[1];
rz(-0.2793667180176756) q[1];
rx(-0.019321231493905928) q[2];
rz(-0.4108777696581236) q[2];
rx(0.0909737443768124) q[3];
rz(-0.0848394051242929) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08093234973888222) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11678259593494232) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2627187378426825) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7716320496593114) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.48126857652709865) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12799546198007072) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3675228674531912) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.27814354491840393) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.22903565595935194) q[3];
cx q[2],q[3];
rx(-0.009070121964547088) q[0];
rz(0.04641759717674049) q[0];
rx(0.23342589273887507) q[1];
rz(-0.07382388412305858) q[1];
rx(-0.02171747420067197) q[2];
rz(-0.3871936839854714) q[2];
rx(0.01889816413952589) q[3];
rz(-0.15450427367175257) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.14372773341377015) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.09246110026847894) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.36874606964587026) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.5772798107983503) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3148175892991472) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07504055191874906) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.33278446873925815) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2213128154676672) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2707949942553226) q[3];
cx q[2],q[3];
rx(0.027335786035929243) q[0];
rz(0.08860369134424408) q[0];
rx(0.11409444255420875) q[1];
rz(0.30138602434760225) q[1];
rx(-0.03339904032804069) q[2];
rz(-0.19876338847452935) q[2];
rx(-0.027165326460919157) q[3];
rz(-0.3013948039468082) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.020729832346864336) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.008032446868934753) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.4393833595800099) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.44057559333581664) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.060910036703917037) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06784662078182174) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.22349823589978218) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.038550819958558004) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13685918333891778) q[3];
cx q[2],q[3];
rx(0.13217447092195625) q[0];
rz(0.050559585500807094) q[0];
rx(0.11977260038574594) q[1];
rz(0.2961929870548964) q[1];
rx(-0.14678219032756323) q[2];
rz(-0.06358666191310079) q[2];
rx(-0.07461472713424733) q[3];
rz(-0.2594992567098126) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.08472481005790981) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03180153603831536) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.22120124143423636) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11565504723422487) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.22073332429315687) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.10607721148939551) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.273926671055478) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.04172455856051439) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07336438479809171) q[3];
cx q[2],q[3];
rx(0.10762696098129675) q[0];
rz(0.004324759954655488) q[0];
rx(0.043516430223112) q[1];
rz(0.13100459177277465) q[1];
rx(-0.08878277928126846) q[2];
rz(-0.1144352206073029) q[2];
rx(-0.1568062469288576) q[3];
rz(-0.12907112450437397) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10309548121802847) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.005730822484482473) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05108424474384434) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.13225980142738866) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.46534888570796556) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04364164824236203) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.250168711989636) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2462837259032912) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2109063745242292) q[3];
cx q[2],q[3];
rx(0.12623772840997824) q[0];
rz(0.01242367470375688) q[0];
rx(-0.1594136126176972) q[1];
rz(-0.1738304071112151) q[1];
rx(-0.019810290019763158) q[2];
rz(-0.20506761417208433) q[2];
rx(-0.19537285619904285) q[3];
rz(-0.0643185107442424) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.12891039286647987) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02428555294335381) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06860214067031675) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.5476686060341845) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6369831901738818) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.028385827081553707) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.21623760369308306) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2171113604539112) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.22200231128908393) q[3];
cx q[2],q[3];
rx(0.15780249723117382) q[0];
rz(-0.05343216239849307) q[0];
rx(-0.3551554742502448) q[1];
rz(-0.11261674338695903) q[1];
rx(0.07404190917832161) q[2];
rz(-0.06098643605046566) q[2];
rx(-0.18052266716207852) q[3];
rz(-0.012875601520558205) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.028942030936140953) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07212210316765856) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.026544065470775865) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.6690595506090707) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7203119215042394) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.24356963402669915) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.020705020007946353) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2733817778884074) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.4473788936424391) q[3];
cx q[2],q[3];
rx(0.18682457586352524) q[0];
rz(0.00027421448969487325) q[0];
rx(-0.3115932621725946) q[1];
rz(-0.05189180788623116) q[1];
rx(0.07826553699497563) q[2];
rz(0.13213896184184573) q[2];
rx(-0.11409512489115614) q[3];
rz(-0.050082998216622765) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.050032328893436714) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2629230389594259) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.017585106698492243) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.7988493096643007) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6389735215545747) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.27085399450078773) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.055735775279099585) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1567498422143614) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.39558655978476936) q[3];
cx q[2],q[3];
rx(0.16695628503402332) q[0];
rz(0.043067720855941606) q[0];
rx(-0.09772002942760512) q[1];
rz(-0.10576577776053785) q[1];
rx(0.06488275097170167) q[2];
rz(0.19140725050593219) q[2];
rx(-0.08948486493100692) q[3];
rz(-0.06279835636393064) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0842696446840445) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.47553950030797554) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01902986757010551) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.8277460479785821) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7435462158418704) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.40050719592897244) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.1620069635885575) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1763726472371728) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05299937353791591) q[3];
cx q[2],q[3];
rx(0.17051959787282503) q[0];
rz(0.04426806821765404) q[0];
rx(-0.008839916699725442) q[1];
rz(-0.19978370597062436) q[1];
rx(-0.02329260683256704) q[2];
rz(0.16742623109679813) q[2];
rx(-0.17178965147826797) q[3];
rz(-0.08503463136033756) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.13877057078778154) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2543509660404456) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11101473580773628) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.9284803047697592) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6616305449493918) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5413723907708451) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.00548863296039947) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.25814564120851785) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.17972884323606672) q[3];
cx q[2],q[3];
rx(0.2908414047804215) q[0];
rz(0.08960416553932858) q[0];
rx(-0.08077586704729323) q[1];
rz(-0.4331130835798927) q[1];
rx(0.03652934847399683) q[2];
rz(0.29366275959024046) q[2];
rx(-0.1379124938487314) q[3];
rz(-0.12139060024141095) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.24467143133761052) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11258325949316512) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.29080361957111234) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.786105100372976) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5352678731907504) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6448976449502534) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.17903214412319277) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2883208739636222) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08648892647276818) q[3];
cx q[2],q[3];
rx(0.5109981146638484) q[0];
rz(0.19574352907489725) q[0];
rx(-0.22407244252330596) q[1];
rz(-0.576625626706354) q[1];
rx(-0.16573681639812532) q[2];
rz(0.2732815188758782) q[2];
rx(-0.05105365445061002) q[3];
rz(-0.12703654782343193) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.24804916389552342) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.030559232971319968) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.46439900125567846) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.515226781480885) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11569872356857447) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6910728187595921) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.21548716286197703) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1460281088620662) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.246094144102062) q[3];
cx q[2],q[3];
rx(0.665111882298291) q[0];
rz(0.2777441763817292) q[0];
rx(-0.3526682638219582) q[1];
rz(-0.6732045762049844) q[1];
rx(-0.36013027126949804) q[2];
rz(0.2536920688708953) q[2];
rx(0.07053569116043795) q[3];
rz(-0.08061940755645917) q[3];