OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.2315980227139822) q[0];
ry(-2.3507236233390376) q[1];
cx q[0],q[1];
ry(-1.4124942846271953) q[0];
ry(-2.17126895107005) q[1];
cx q[0],q[1];
ry(-2.341148345673238) q[0];
ry(1.84515134851215) q[2];
cx q[0],q[2];
ry(-2.358162911280967) q[0];
ry(1.1070322329255389) q[2];
cx q[0],q[2];
ry(-2.707661885886436) q[0];
ry(1.216679489379696) q[3];
cx q[0],q[3];
ry(1.564112860667366) q[0];
ry(0.9125979393953575) q[3];
cx q[0],q[3];
ry(-0.5491472556503476) q[0];
ry(-1.563782390881347) q[4];
cx q[0],q[4];
ry(0.032070062779884445) q[0];
ry(-1.0183953267819088) q[4];
cx q[0],q[4];
ry(-0.810596705033651) q[0];
ry(-0.3093634909056205) q[5];
cx q[0],q[5];
ry(-2.5783001702153587) q[0];
ry(1.100644587912285) q[5];
cx q[0],q[5];
ry(-2.9412641292886215) q[0];
ry(0.8767738989136834) q[6];
cx q[0],q[6];
ry(-1.5201188163991328) q[0];
ry(-2.9228597851708913) q[6];
cx q[0],q[6];
ry(-0.1809649444738275) q[0];
ry(1.1856637998195099) q[7];
cx q[0],q[7];
ry(-0.21182615389715664) q[0];
ry(1.7986358588677627) q[7];
cx q[0],q[7];
ry(0.31466716709548687) q[1];
ry(-1.148237783835434) q[2];
cx q[1],q[2];
ry(-0.9596311934871951) q[1];
ry(-0.08485432125746346) q[2];
cx q[1],q[2];
ry(2.5958710622186296) q[1];
ry(2.0544911412464995) q[3];
cx q[1],q[3];
ry(-1.5384212265060688) q[1];
ry(-0.28661513968191077) q[3];
cx q[1],q[3];
ry(1.165423965426748) q[1];
ry(-1.1192685509307791) q[4];
cx q[1],q[4];
ry(-0.6378123877860107) q[1];
ry(-2.666321795093107) q[4];
cx q[1],q[4];
ry(-0.9299950281710113) q[1];
ry(0.7582910729637379) q[5];
cx q[1],q[5];
ry(0.09832513023419587) q[1];
ry(3.115894127071086) q[5];
cx q[1],q[5];
ry(0.888137639010119) q[1];
ry(2.9860384214394573) q[6];
cx q[1],q[6];
ry(-1.8743507410201554) q[1];
ry(2.6882397448557076) q[6];
cx q[1],q[6];
ry(2.1591695847298915) q[1];
ry(-1.442972960218654) q[7];
cx q[1],q[7];
ry(-2.8284296947772902) q[1];
ry(-2.4761953661293448) q[7];
cx q[1],q[7];
ry(-0.48127470859993604) q[2];
ry(1.3544960417748102) q[3];
cx q[2],q[3];
ry(-0.6867683513085909) q[2];
ry(-3.082375960918359) q[3];
cx q[2],q[3];
ry(2.4139957948282587) q[2];
ry(-2.199373504872737) q[4];
cx q[2],q[4];
ry(2.959712337224062) q[2];
ry(-2.6642742718873307) q[4];
cx q[2],q[4];
ry(2.979702206341063) q[2];
ry(-0.3738400051457358) q[5];
cx q[2],q[5];
ry(-1.4727918718675408) q[2];
ry(1.5098812128962689) q[5];
cx q[2],q[5];
ry(-1.9278980538868946) q[2];
ry(1.649250436458414) q[6];
cx q[2],q[6];
ry(-1.4289909647463395) q[2];
ry(2.608131380636953) q[6];
cx q[2],q[6];
ry(2.8779663198643024) q[2];
ry(-1.6137808607653754) q[7];
cx q[2],q[7];
ry(-1.7363933301142505) q[2];
ry(0.8951889263910164) q[7];
cx q[2],q[7];
ry(-2.7769493394712246) q[3];
ry(1.6166316817104476) q[4];
cx q[3],q[4];
ry(-2.86973875993534) q[3];
ry(-1.4431669888847312) q[4];
cx q[3],q[4];
ry(0.4683078843698594) q[3];
ry(0.1704586205456422) q[5];
cx q[3],q[5];
ry(2.652957185989264) q[3];
ry(0.8519851854020635) q[5];
cx q[3],q[5];
ry(0.6221017057775255) q[3];
ry(1.3028728194931354) q[6];
cx q[3],q[6];
ry(-2.6104075806383586) q[3];
ry(1.1468816376736966) q[6];
cx q[3],q[6];
ry(-1.2820897103839926) q[3];
ry(3.1169357593666915) q[7];
cx q[3],q[7];
ry(0.8809916745398354) q[3];
ry(-2.400701307015481) q[7];
cx q[3],q[7];
ry(-1.1786830558774881) q[4];
ry(-2.730447661767657) q[5];
cx q[4],q[5];
ry(2.2084703347813988) q[4];
ry(-0.7390940785614051) q[5];
cx q[4],q[5];
ry(-1.0228441982228753) q[4];
ry(2.979519624613865) q[6];
cx q[4],q[6];
ry(-0.9672258054538363) q[4];
ry(-2.2475991797529966) q[6];
cx q[4],q[6];
ry(0.6797446251927637) q[4];
ry(0.2739973298649393) q[7];
cx q[4],q[7];
ry(-2.91580147291885) q[4];
ry(-1.2765820793063858) q[7];
cx q[4],q[7];
ry(3.138428776573523) q[5];
ry(-1.457113846357961) q[6];
cx q[5],q[6];
ry(-1.970909637306867) q[5];
ry(2.5409301446913943) q[6];
cx q[5],q[6];
ry(1.5744822818724789) q[5];
ry(1.9072539110717495) q[7];
cx q[5],q[7];
ry(-2.0538206852836316) q[5];
ry(-1.974138289995973) q[7];
cx q[5],q[7];
ry(1.4951191344653516) q[6];
ry(-2.522489507731285) q[7];
cx q[6],q[7];
ry(1.4189539011590604) q[6];
ry(-0.4561192519015771) q[7];
cx q[6],q[7];
ry(1.8170869091692061) q[0];
ry(0.18069023032063658) q[1];
cx q[0],q[1];
ry(1.8864143439873367) q[0];
ry(0.2786196874040181) q[1];
cx q[0],q[1];
ry(-1.2849279032029484) q[0];
ry(0.3241500681952543) q[2];
cx q[0],q[2];
ry(-1.911241944668312) q[0];
ry(3.0344572663717675) q[2];
cx q[0],q[2];
ry(-0.2431545767297424) q[0];
ry(2.3943258705484376) q[3];
cx q[0],q[3];
ry(-1.5149053496830112) q[0];
ry(-1.4075688605477845) q[3];
cx q[0],q[3];
ry(2.3279911584605757) q[0];
ry(-2.1955093835100845) q[4];
cx q[0],q[4];
ry(-1.7568162742230018) q[0];
ry(-0.09819588655984023) q[4];
cx q[0],q[4];
ry(1.6279180373398408) q[0];
ry(-1.0212853186950808) q[5];
cx q[0],q[5];
ry(0.9831131851068929) q[0];
ry(2.500073721150986) q[5];
cx q[0],q[5];
ry(-1.9795236084571506) q[0];
ry(1.8668298772897394) q[6];
cx q[0],q[6];
ry(2.153316547117345) q[0];
ry(-1.8471034541678022) q[6];
cx q[0],q[6];
ry(1.105923593282909) q[0];
ry(2.873951952902357) q[7];
cx q[0],q[7];
ry(-0.32528932961292223) q[0];
ry(-2.1951364210558353) q[7];
cx q[0],q[7];
ry(-1.1898455762818954) q[1];
ry(-2.416833869587513) q[2];
cx q[1],q[2];
ry(2.569177415205264) q[1];
ry(-1.7131647192974901) q[2];
cx q[1],q[2];
ry(0.18197128969867404) q[1];
ry(1.1725478835674081) q[3];
cx q[1],q[3];
ry(-1.8771859110162474) q[1];
ry(2.1616232107919826) q[3];
cx q[1],q[3];
ry(-0.6853677941363587) q[1];
ry(1.031887062758492) q[4];
cx q[1],q[4];
ry(1.9578231845802012) q[1];
ry(1.2764569533087913) q[4];
cx q[1],q[4];
ry(0.6441396546569444) q[1];
ry(2.3902632423757995) q[5];
cx q[1],q[5];
ry(3.030626925356076) q[1];
ry(2.2601561616716084) q[5];
cx q[1],q[5];
ry(-2.3104924726511036) q[1];
ry(1.455131833410027) q[6];
cx q[1],q[6];
ry(1.8083895929090406) q[1];
ry(-0.5958230316656401) q[6];
cx q[1],q[6];
ry(-0.21694949543909983) q[1];
ry(-1.4560350121169625) q[7];
cx q[1],q[7];
ry(-2.5849368123514345) q[1];
ry(1.9943587446933562) q[7];
cx q[1],q[7];
ry(2.030741819628825) q[2];
ry(0.9684998336339811) q[3];
cx q[2],q[3];
ry(-3.1045147464785914) q[2];
ry(0.9041998199883299) q[3];
cx q[2],q[3];
ry(2.163719959555507) q[2];
ry(-1.0379767447634096) q[4];
cx q[2],q[4];
ry(3.0676479211449057) q[2];
ry(-0.7380520221314852) q[4];
cx q[2],q[4];
ry(1.0866142232294584) q[2];
ry(-1.915797927068219) q[5];
cx q[2],q[5];
ry(-1.7483362029843545) q[2];
ry(-1.080767073344258) q[5];
cx q[2],q[5];
ry(2.0936534511972917) q[2];
ry(-2.949987244483093) q[6];
cx q[2],q[6];
ry(-1.5898836312809541) q[2];
ry(2.20892497019532) q[6];
cx q[2],q[6];
ry(1.8452248369217377) q[2];
ry(0.8780669115663723) q[7];
cx q[2],q[7];
ry(-0.7858097791182083) q[2];
ry(0.1152600717886448) q[7];
cx q[2],q[7];
ry(1.6738280642438885) q[3];
ry(1.855164124173209) q[4];
cx q[3],q[4];
ry(-3.083117564261288) q[3];
ry(0.8474110273258662) q[4];
cx q[3],q[4];
ry(1.084385929791856) q[3];
ry(-2.958473124965547) q[5];
cx q[3],q[5];
ry(-1.466680856709929) q[3];
ry(-2.8759812138207423) q[5];
cx q[3],q[5];
ry(1.1934955930902815) q[3];
ry(-0.61754660628071) q[6];
cx q[3],q[6];
ry(2.1869962006003574) q[3];
ry(2.7906196547321356) q[6];
cx q[3],q[6];
ry(0.4449873913958813) q[3];
ry(1.1064091905156328) q[7];
cx q[3],q[7];
ry(-1.9737026480101598) q[3];
ry(-0.30521225813199493) q[7];
cx q[3],q[7];
ry(-1.545512990113914) q[4];
ry(0.4476817876669523) q[5];
cx q[4],q[5];
ry(-0.3851731399183306) q[4];
ry(-2.3763519464150056) q[5];
cx q[4],q[5];
ry(2.495450146897588) q[4];
ry(-2.3126897233027743) q[6];
cx q[4],q[6];
ry(2.1845912437440234) q[4];
ry(-1.391497562415888) q[6];
cx q[4],q[6];
ry(-1.8230957824552834) q[4];
ry(0.9120899721971952) q[7];
cx q[4],q[7];
ry(-0.8229061949094384) q[4];
ry(-1.2632109429340908) q[7];
cx q[4],q[7];
ry(-2.293470236101146) q[5];
ry(-2.980530671598315) q[6];
cx q[5],q[6];
ry(-2.516582599964578) q[5];
ry(0.8152136850145304) q[6];
cx q[5],q[6];
ry(0.7352799545285098) q[5];
ry(-0.7097361236822367) q[7];
cx q[5],q[7];
ry(0.6008696644511431) q[5];
ry(-1.9645606308805754) q[7];
cx q[5],q[7];
ry(-2.6725552957661285) q[6];
ry(2.550689152723767) q[7];
cx q[6],q[7];
ry(-3.0272700885153148) q[6];
ry(-2.771533698710374) q[7];
cx q[6],q[7];
ry(1.4453889561244797) q[0];
ry(-1.1600207465359151) q[1];
cx q[0],q[1];
ry(0.7188115945544586) q[0];
ry(2.3157752548064714) q[1];
cx q[0],q[1];
ry(2.0499708604812907) q[0];
ry(-2.0037676957166415) q[2];
cx q[0],q[2];
ry(-0.7447304810296911) q[0];
ry(2.6616090630370186) q[2];
cx q[0],q[2];
ry(0.5031219639957425) q[0];
ry(-1.306653319407214) q[3];
cx q[0],q[3];
ry(-2.070033814985341) q[0];
ry(-0.6841201990137007) q[3];
cx q[0],q[3];
ry(1.1702729000957666) q[0];
ry(-2.4281001913412807) q[4];
cx q[0],q[4];
ry(-2.9382589296443666) q[0];
ry(1.8809816371448411) q[4];
cx q[0],q[4];
ry(-1.415626942922965) q[0];
ry(-2.1596920209224884) q[5];
cx q[0],q[5];
ry(1.0316412459177646) q[0];
ry(0.8932197562390943) q[5];
cx q[0],q[5];
ry(2.9422328501857056) q[0];
ry(-1.4140186934840528) q[6];
cx q[0],q[6];
ry(-0.09627223148876318) q[0];
ry(1.301334803076613) q[6];
cx q[0],q[6];
ry(-1.7142635887480022) q[0];
ry(-2.024936690999005) q[7];
cx q[0],q[7];
ry(1.65949150320023) q[0];
ry(1.3401085477093284) q[7];
cx q[0],q[7];
ry(2.3937991097578655) q[1];
ry(-0.4529712919380575) q[2];
cx q[1],q[2];
ry(0.6749535478703503) q[1];
ry(-1.6747333043999224) q[2];
cx q[1],q[2];
ry(-1.7663255240251572) q[1];
ry(1.8334772455898378) q[3];
cx q[1],q[3];
ry(0.8199678705574858) q[1];
ry(1.8731705837680694) q[3];
cx q[1],q[3];
ry(0.3173753794681285) q[1];
ry(0.472003723770551) q[4];
cx q[1],q[4];
ry(-2.265194365420804) q[1];
ry(-1.6174326464015287) q[4];
cx q[1],q[4];
ry(1.6320327023919738) q[1];
ry(0.489296026379499) q[5];
cx q[1],q[5];
ry(1.1437324149088808) q[1];
ry(0.45622054588611566) q[5];
cx q[1],q[5];
ry(1.714232393829631) q[1];
ry(-2.055580373124431) q[6];
cx q[1],q[6];
ry(2.6702222966760276) q[1];
ry(-0.17295285293815518) q[6];
cx q[1],q[6];
ry(0.84649539950473) q[1];
ry(-2.1929312190839227) q[7];
cx q[1],q[7];
ry(1.6079590852183614) q[1];
ry(-2.2492861223937295) q[7];
cx q[1],q[7];
ry(-1.9118374639348623) q[2];
ry(-1.176438920832342) q[3];
cx q[2],q[3];
ry(0.5680678212458915) q[2];
ry(2.872909745629011) q[3];
cx q[2],q[3];
ry(1.0932516613861765) q[2];
ry(2.6994563031201344) q[4];
cx q[2],q[4];
ry(1.1470867820786257) q[2];
ry(-1.0435577014050006) q[4];
cx q[2],q[4];
ry(0.8290620244450044) q[2];
ry(-0.4317604288168644) q[5];
cx q[2],q[5];
ry(-2.214910395016807) q[2];
ry(-1.459795196186444) q[5];
cx q[2],q[5];
ry(2.04218243332702) q[2];
ry(-1.901019549162224) q[6];
cx q[2],q[6];
ry(0.3806961788060731) q[2];
ry(1.3907676408893632) q[6];
cx q[2],q[6];
ry(0.63728514002084) q[2];
ry(-2.256281770825423) q[7];
cx q[2],q[7];
ry(1.6267478861984597) q[2];
ry(-1.401284926790006) q[7];
cx q[2],q[7];
ry(2.380084403750456) q[3];
ry(1.7973258925484075) q[4];
cx q[3],q[4];
ry(-1.293794620526943) q[3];
ry(1.8092272299786596) q[4];
cx q[3],q[4];
ry(1.7107544963212664) q[3];
ry(1.3021324086011188) q[5];
cx q[3],q[5];
ry(0.27951110208910973) q[3];
ry(0.11279939905734943) q[5];
cx q[3],q[5];
ry(-0.2812408202329397) q[3];
ry(-2.531138915460076) q[6];
cx q[3],q[6];
ry(1.0420349708773342) q[3];
ry(-1.6293415814725263) q[6];
cx q[3],q[6];
ry(2.274219437921798) q[3];
ry(-1.2285997307190735) q[7];
cx q[3],q[7];
ry(1.6562156952711118) q[3];
ry(2.4339144475698835) q[7];
cx q[3],q[7];
ry(-2.7489840051784964) q[4];
ry(-2.9548677086505477) q[5];
cx q[4],q[5];
ry(2.0372202159397705) q[4];
ry(-1.6252707080695399) q[5];
cx q[4],q[5];
ry(2.4218418366220917) q[4];
ry(2.4217634502393475) q[6];
cx q[4],q[6];
ry(-0.2436041629389363) q[4];
ry(0.0704744831037955) q[6];
cx q[4],q[6];
ry(-2.7765365408621205) q[4];
ry(-2.966508383028988) q[7];
cx q[4],q[7];
ry(0.3218147744248867) q[4];
ry(0.2979945240937534) q[7];
cx q[4],q[7];
ry(1.5835276949830783) q[5];
ry(-2.898621299008761) q[6];
cx q[5],q[6];
ry(-2.9035099687178407) q[5];
ry(0.6309004704150446) q[6];
cx q[5],q[6];
ry(0.7506873209299696) q[5];
ry(2.057546489446578) q[7];
cx q[5],q[7];
ry(0.07546127329883277) q[5];
ry(-1.088250890242867) q[7];
cx q[5],q[7];
ry(-1.713136146437411) q[6];
ry(-1.7873805200286945) q[7];
cx q[6],q[7];
ry(1.3517935165233856) q[6];
ry(-0.43559595536006773) q[7];
cx q[6],q[7];
ry(0.43328911258756125) q[0];
ry(-2.3341143436403384) q[1];
cx q[0],q[1];
ry(3.130339600964191) q[0];
ry(-2.4884505369511003) q[1];
cx q[0],q[1];
ry(1.3782241735110328) q[0];
ry(-1.220553710448823) q[2];
cx q[0],q[2];
ry(-2.3072806885926402) q[0];
ry(-2.6352849765730864) q[2];
cx q[0],q[2];
ry(0.6308396002496351) q[0];
ry(0.7019352926069109) q[3];
cx q[0],q[3];
ry(0.16028953511080907) q[0];
ry(2.3772531187177695) q[3];
cx q[0],q[3];
ry(-0.04363101366481912) q[0];
ry(-2.7372263919942776) q[4];
cx q[0],q[4];
ry(-1.8836677548342466) q[0];
ry(-1.3408091278321428) q[4];
cx q[0],q[4];
ry(2.6152559766619508) q[0];
ry(2.259645408079997) q[5];
cx q[0],q[5];
ry(-2.1455350016844954) q[0];
ry(0.23877292361033822) q[5];
cx q[0],q[5];
ry(1.3335382730596812) q[0];
ry(0.35276525163776995) q[6];
cx q[0],q[6];
ry(2.086152927989306) q[0];
ry(-2.821359509459934) q[6];
cx q[0],q[6];
ry(-2.0587420151990665) q[0];
ry(1.5801202974960449) q[7];
cx q[0],q[7];
ry(-1.9488972172403445) q[0];
ry(0.6860498075919924) q[7];
cx q[0],q[7];
ry(0.6097756246302902) q[1];
ry(-3.0968136682757064) q[2];
cx q[1],q[2];
ry(-2.4604246272919883) q[1];
ry(1.0127144851856968) q[2];
cx q[1],q[2];
ry(2.0647027983636925) q[1];
ry(-2.510820391499504) q[3];
cx q[1],q[3];
ry(2.209908759867098) q[1];
ry(0.8347481759060349) q[3];
cx q[1],q[3];
ry(-0.7639724087640084) q[1];
ry(1.653491261587894) q[4];
cx q[1],q[4];
ry(2.6848994623071714) q[1];
ry(-0.6494703976976209) q[4];
cx q[1],q[4];
ry(-2.1759986733438588) q[1];
ry(-0.6392330654443226) q[5];
cx q[1],q[5];
ry(0.9957176752093724) q[1];
ry(-1.7029663184909964) q[5];
cx q[1],q[5];
ry(-1.6562930507678768) q[1];
ry(2.4690580665682127) q[6];
cx q[1],q[6];
ry(-0.4877656772715824) q[1];
ry(0.36020035155380903) q[6];
cx q[1],q[6];
ry(3.029941597317183) q[1];
ry(-0.5673973130631081) q[7];
cx q[1],q[7];
ry(1.8252285665858607) q[1];
ry(1.5478167818653306) q[7];
cx q[1],q[7];
ry(0.36945114279389396) q[2];
ry(2.0607264741525064) q[3];
cx q[2],q[3];
ry(-1.4729456109642625) q[2];
ry(3.0418997373454695) q[3];
cx q[2],q[3];
ry(1.1590367981839211) q[2];
ry(0.6140312013717653) q[4];
cx q[2],q[4];
ry(2.3676989286245504) q[2];
ry(-0.9862967893617789) q[4];
cx q[2],q[4];
ry(-1.5907840587949778) q[2];
ry(1.682464587184999) q[5];
cx q[2],q[5];
ry(1.864431803464531) q[2];
ry(1.1111533531412434) q[5];
cx q[2],q[5];
ry(1.3385289086382315) q[2];
ry(2.4214648483806176) q[6];
cx q[2],q[6];
ry(3.0545724790638378) q[2];
ry(-2.054568726393679) q[6];
cx q[2],q[6];
ry(2.358912514274234) q[2];
ry(-2.1881449905639583) q[7];
cx q[2],q[7];
ry(0.18112238845829332) q[2];
ry(0.4557047656053932) q[7];
cx q[2],q[7];
ry(-1.6351638313927515) q[3];
ry(-1.6318281453345218) q[4];
cx q[3],q[4];
ry(0.7862986005511319) q[3];
ry(2.62063819079614) q[4];
cx q[3],q[4];
ry(-2.3258250487784338) q[3];
ry(-2.5037834138085673) q[5];
cx q[3],q[5];
ry(-1.1122552328993933) q[3];
ry(1.0480488541574609) q[5];
cx q[3],q[5];
ry(1.4449353268931713) q[3];
ry(1.7587678038388308) q[6];
cx q[3],q[6];
ry(-2.281673794448277) q[3];
ry(-0.4304895825877759) q[6];
cx q[3],q[6];
ry(0.22987455337867144) q[3];
ry(-1.078201406740723) q[7];
cx q[3],q[7];
ry(-0.614166728856301) q[3];
ry(-0.8666853164749347) q[7];
cx q[3],q[7];
ry(0.283729176755912) q[4];
ry(2.3109939745409767) q[5];
cx q[4],q[5];
ry(1.7739088996691141) q[4];
ry(0.6368262921577026) q[5];
cx q[4],q[5];
ry(2.257309540485433) q[4];
ry(-0.47928155471135514) q[6];
cx q[4],q[6];
ry(0.6695105360996474) q[4];
ry(-2.498728471984326) q[6];
cx q[4],q[6];
ry(0.6011405878688924) q[4];
ry(-0.013047067529216706) q[7];
cx q[4],q[7];
ry(2.3514615453691747) q[4];
ry(-0.2119892219817121) q[7];
cx q[4],q[7];
ry(2.2527003778598793) q[5];
ry(0.9925359527530421) q[6];
cx q[5],q[6];
ry(-1.533465251205406) q[5];
ry(1.474405658147023) q[6];
cx q[5],q[6];
ry(-3.114221836555988) q[5];
ry(1.8212418080631068) q[7];
cx q[5],q[7];
ry(1.0739911061083998) q[5];
ry(3.015838024813415) q[7];
cx q[5],q[7];
ry(-1.0902250847623245) q[6];
ry(-3.056152210256139) q[7];
cx q[6],q[7];
ry(-0.10873039109314245) q[6];
ry(-1.5529148664160806) q[7];
cx q[6],q[7];
ry(-0.9478907860265365) q[0];
ry(-0.6571961091468612) q[1];
cx q[0],q[1];
ry(2.4897756076153574) q[0];
ry(-1.5242682931865426) q[1];
cx q[0],q[1];
ry(2.9031172458222443) q[0];
ry(0.051903097268739096) q[2];
cx q[0],q[2];
ry(-1.7393667547988132) q[0];
ry(2.5705890366854587) q[2];
cx q[0],q[2];
ry(0.7294607621728391) q[0];
ry(-0.20646414664612983) q[3];
cx q[0],q[3];
ry(1.8482671444717982) q[0];
ry(2.07680605019815) q[3];
cx q[0],q[3];
ry(1.612175557003818) q[0];
ry(-2.6206965674940546) q[4];
cx q[0],q[4];
ry(0.9448323463285304) q[0];
ry(-0.8810711257190194) q[4];
cx q[0],q[4];
ry(-1.0492659719471749) q[0];
ry(-0.7948433627770948) q[5];
cx q[0],q[5];
ry(2.2750577783769197) q[0];
ry(-1.9078790044663752) q[5];
cx q[0],q[5];
ry(2.77940331414581) q[0];
ry(-2.03428673229545) q[6];
cx q[0],q[6];
ry(-1.8602715185722516) q[0];
ry(-2.5743775486183407) q[6];
cx q[0],q[6];
ry(-1.3737861192753995) q[0];
ry(0.9573648607423774) q[7];
cx q[0],q[7];
ry(-1.3231987195286026) q[0];
ry(-0.7492748435398688) q[7];
cx q[0],q[7];
ry(1.773907658697615) q[1];
ry(-1.0618961839225707) q[2];
cx q[1],q[2];
ry(3.048726235880023) q[1];
ry(-1.5087496318209874) q[2];
cx q[1],q[2];
ry(-0.723187521819451) q[1];
ry(1.2296398957439152) q[3];
cx q[1],q[3];
ry(0.336450346501688) q[1];
ry(-2.9740503628857557) q[3];
cx q[1],q[3];
ry(2.7354848108377627) q[1];
ry(-1.3392621428577405) q[4];
cx q[1],q[4];
ry(-1.648564923750705) q[1];
ry(-2.6173974059374703) q[4];
cx q[1],q[4];
ry(0.9985509194894115) q[1];
ry(-2.86270386147237) q[5];
cx q[1],q[5];
ry(-2.9754795819927615) q[1];
ry(2.1130138689027955) q[5];
cx q[1],q[5];
ry(0.2627636618648907) q[1];
ry(-2.627424397426565) q[6];
cx q[1],q[6];
ry(2.9351900338829617) q[1];
ry(0.39243453473271195) q[6];
cx q[1],q[6];
ry(-0.060966316859472514) q[1];
ry(-1.569530947489623) q[7];
cx q[1],q[7];
ry(-0.06672743866225411) q[1];
ry(2.959765719973546) q[7];
cx q[1],q[7];
ry(-0.131566052859017) q[2];
ry(-0.3006861961488141) q[3];
cx q[2],q[3];
ry(0.7707763702172431) q[2];
ry(-1.7656075612686983) q[3];
cx q[2],q[3];
ry(1.7722681782262297) q[2];
ry(-0.06855736928502054) q[4];
cx q[2],q[4];
ry(-1.5424776576023043) q[2];
ry(2.557745280677347) q[4];
cx q[2],q[4];
ry(0.44489463497692405) q[2];
ry(0.6877977433330462) q[5];
cx q[2],q[5];
ry(-0.6906920446821365) q[2];
ry(-0.6955675386903769) q[5];
cx q[2],q[5];
ry(1.554663102014806) q[2];
ry(-1.64351272752956) q[6];
cx q[2],q[6];
ry(-1.6897191854432005) q[2];
ry(0.8120164591305974) q[6];
cx q[2],q[6];
ry(1.4699935847111656) q[2];
ry(1.1158906782067408) q[7];
cx q[2],q[7];
ry(-0.6588623249485996) q[2];
ry(1.2861759999614255) q[7];
cx q[2],q[7];
ry(-2.895247342829279) q[3];
ry(-2.1659292434070023) q[4];
cx q[3],q[4];
ry(0.6390915853844161) q[3];
ry(0.24463406384733952) q[4];
cx q[3],q[4];
ry(-0.05768928276224283) q[3];
ry(0.03142136572238208) q[5];
cx q[3],q[5];
ry(-1.7211585961916533) q[3];
ry(-2.3374725969024968) q[5];
cx q[3],q[5];
ry(0.9649875467746103) q[3];
ry(1.1092784520507513) q[6];
cx q[3],q[6];
ry(-0.8059841683582043) q[3];
ry(2.7791213788464337) q[6];
cx q[3],q[6];
ry(-1.9063504295435652) q[3];
ry(-0.10410393800882023) q[7];
cx q[3],q[7];
ry(2.308972119620514) q[3];
ry(2.824820448857665) q[7];
cx q[3],q[7];
ry(1.6711864135587642) q[4];
ry(1.2671195644504967) q[5];
cx q[4],q[5];
ry(-0.3628504634578613) q[4];
ry(3.12898422115045) q[5];
cx q[4],q[5];
ry(-2.443967792155695) q[4];
ry(0.779699260004992) q[6];
cx q[4],q[6];
ry(2.8326175437343015) q[4];
ry(-2.6416979635976237) q[6];
cx q[4],q[6];
ry(2.139455976067206) q[4];
ry(-1.8283425174744055) q[7];
cx q[4],q[7];
ry(2.6176856629583276) q[4];
ry(1.2344534388743487) q[7];
cx q[4],q[7];
ry(0.8048398634882354) q[5];
ry(1.385267268797648) q[6];
cx q[5],q[6];
ry(0.00392020855165498) q[5];
ry(0.015669482056554707) q[6];
cx q[5],q[6];
ry(-0.6047595657558398) q[5];
ry(-0.31473133642557055) q[7];
cx q[5],q[7];
ry(2.2686906253165997) q[5];
ry(-2.48440298482238) q[7];
cx q[5],q[7];
ry(-2.187033836185102) q[6];
ry(-1.2123949523344857) q[7];
cx q[6],q[7];
ry(-2.7504902575076717) q[6];
ry(2.115630787557981) q[7];
cx q[6],q[7];
ry(-3.14073338128821) q[0];
ry(-0.913154831366688) q[1];
ry(-1.7027155677810555) q[2];
ry(2.3371279089129744) q[3];
ry(-2.3473444211721457) q[4];
ry(-1.773155193287954) q[5];
ry(-1.83631687327797) q[6];
ry(-1.549198928059397) q[7];