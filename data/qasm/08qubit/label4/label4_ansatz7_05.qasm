OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.3453535686922415) q[0];
ry(-0.17132512107585463) q[1];
cx q[0],q[1];
ry(0.06235671963893719) q[0];
ry(-2.6146004414271147) q[1];
cx q[0],q[1];
ry(-1.7008502680860067) q[0];
ry(-1.7388424772604172) q[2];
cx q[0],q[2];
ry(1.4091179103905143) q[0];
ry(2.556347270735686) q[2];
cx q[0],q[2];
ry(-2.424606676244223) q[0];
ry(2.218956554836878) q[3];
cx q[0],q[3];
ry(-1.4461921600032088) q[0];
ry(-2.8584493058820346) q[3];
cx q[0],q[3];
ry(1.3488258501254158) q[0];
ry(-0.43259978987388686) q[4];
cx q[0],q[4];
ry(0.4411176758477282) q[0];
ry(-0.8431226193207327) q[4];
cx q[0],q[4];
ry(-0.7562091138735711) q[0];
ry(-2.6458626278960398) q[5];
cx q[0],q[5];
ry(1.9591604935549656) q[0];
ry(-2.1701110738329987) q[5];
cx q[0],q[5];
ry(1.393310771781662) q[0];
ry(-2.2492800915773685) q[6];
cx q[0],q[6];
ry(2.4865308701364324) q[0];
ry(-2.301130532306048) q[6];
cx q[0],q[6];
ry(-1.7684783994846986) q[0];
ry(-2.256836452312121) q[7];
cx q[0],q[7];
ry(1.1876091808874918) q[0];
ry(2.1918293824656905) q[7];
cx q[0],q[7];
ry(1.6754785043223483) q[1];
ry(2.1510693087358197) q[2];
cx q[1],q[2];
ry(1.619201679313186) q[1];
ry(0.5085678420638456) q[2];
cx q[1],q[2];
ry(0.5305638421654626) q[1];
ry(-1.6756361312314447) q[3];
cx q[1],q[3];
ry(2.298579120541529) q[1];
ry(-1.259168979213631) q[3];
cx q[1],q[3];
ry(-1.099860729216375) q[1];
ry(-0.453338782499924) q[4];
cx q[1],q[4];
ry(2.979594212215839) q[1];
ry(-2.1124771944482994) q[4];
cx q[1],q[4];
ry(-1.3298882672609476) q[1];
ry(-2.6587814039798596) q[5];
cx q[1],q[5];
ry(-2.32235115358965) q[1];
ry(-2.7315368953956973) q[5];
cx q[1],q[5];
ry(0.7683237812906695) q[1];
ry(-2.419717317646121) q[6];
cx q[1],q[6];
ry(-3.0411668615588914) q[1];
ry(-2.592778221438033) q[6];
cx q[1],q[6];
ry(2.580689506817872) q[1];
ry(-1.9588729700634273) q[7];
cx q[1],q[7];
ry(-2.92281505069153) q[1];
ry(1.408122312232493) q[7];
cx q[1],q[7];
ry(2.7306518684701353) q[2];
ry(-1.7782637235371448) q[3];
cx q[2],q[3];
ry(-2.291776305504087) q[2];
ry(0.45620101535104673) q[3];
cx q[2],q[3];
ry(0.7997198013370255) q[2];
ry(1.1835470855017902) q[4];
cx q[2],q[4];
ry(0.5251526452531943) q[2];
ry(2.4822226762952297) q[4];
cx q[2],q[4];
ry(1.6965202803355277) q[2];
ry(-1.008886105801908) q[5];
cx q[2],q[5];
ry(-0.14921356964698712) q[2];
ry(2.3703321425438135) q[5];
cx q[2],q[5];
ry(-2.2657447376139785) q[2];
ry(-1.3122205924931336) q[6];
cx q[2],q[6];
ry(-0.981777843682603) q[2];
ry(-3.0995277524305065) q[6];
cx q[2],q[6];
ry(0.054877471506834564) q[2];
ry(0.4792253064634142) q[7];
cx q[2],q[7];
ry(0.8559508175280673) q[2];
ry(1.6432060863720974) q[7];
cx q[2],q[7];
ry(-3.0711006357727966) q[3];
ry(-1.8428613854055698) q[4];
cx q[3],q[4];
ry(2.796293112009259) q[3];
ry(-0.6801668813443706) q[4];
cx q[3],q[4];
ry(2.9530771461805942) q[3];
ry(1.766810036547683) q[5];
cx q[3],q[5];
ry(-0.6667561307130019) q[3];
ry(2.373633824775769) q[5];
cx q[3],q[5];
ry(0.7417903075384142) q[3];
ry(-3.0829676009099076) q[6];
cx q[3],q[6];
ry(-2.118619496757874) q[3];
ry(-0.4127764658835238) q[6];
cx q[3],q[6];
ry(1.8448736950586984) q[3];
ry(3.008075497385395) q[7];
cx q[3],q[7];
ry(-1.8805884436090294) q[3];
ry(0.6574312642185474) q[7];
cx q[3],q[7];
ry(-2.102397529328899) q[4];
ry(-3.0542146979117075) q[5];
cx q[4],q[5];
ry(-2.381243410588184) q[4];
ry(1.8898023083624507) q[5];
cx q[4],q[5];
ry(-0.04463035594850962) q[4];
ry(0.9511476735800656) q[6];
cx q[4],q[6];
ry(-0.9805445957795564) q[4];
ry(2.4045885211878684) q[6];
cx q[4],q[6];
ry(1.543187089597817) q[4];
ry(-2.143688387541337) q[7];
cx q[4],q[7];
ry(2.654744615933822) q[4];
ry(-0.9355248861710405) q[7];
cx q[4],q[7];
ry(0.6149735901094493) q[5];
ry(1.296181494263486) q[6];
cx q[5],q[6];
ry(-1.933788172766461) q[5];
ry(2.46893610988658) q[6];
cx q[5],q[6];
ry(1.235955941430614) q[5];
ry(1.7752703414562223) q[7];
cx q[5],q[7];
ry(0.9322106818286429) q[5];
ry(-3.081804152565797) q[7];
cx q[5],q[7];
ry(1.2448694637500877) q[6];
ry(-3.1345065045846785) q[7];
cx q[6],q[7];
ry(-0.09624602250877355) q[6];
ry(-2.920227283620915) q[7];
cx q[6],q[7];
ry(-0.085074654219454) q[0];
ry(-1.7909207849066604) q[1];
cx q[0],q[1];
ry(1.22024114639349) q[0];
ry(-2.7551803654620777) q[1];
cx q[0],q[1];
ry(1.9185225553760352) q[0];
ry(2.0757634399786564) q[2];
cx q[0],q[2];
ry(-0.7510897253336736) q[0];
ry(2.731410115638052) q[2];
cx q[0],q[2];
ry(2.980492688862991) q[0];
ry(1.5234119698422788) q[3];
cx q[0],q[3];
ry(-2.7334313890232345) q[0];
ry(-3.020561709005523) q[3];
cx q[0],q[3];
ry(-1.6322265500119173) q[0];
ry(-1.3580989195657232) q[4];
cx q[0],q[4];
ry(-2.3143121503110584) q[0];
ry(0.010540278565119188) q[4];
cx q[0],q[4];
ry(0.5267177203855691) q[0];
ry(-2.6703145189447755) q[5];
cx q[0],q[5];
ry(-0.06901213501423809) q[0];
ry(-3.031865952463151) q[5];
cx q[0],q[5];
ry(-2.726889555732302) q[0];
ry(-2.416587236673308) q[6];
cx q[0],q[6];
ry(-3.0241655889045527) q[0];
ry(0.6428992630066892) q[6];
cx q[0],q[6];
ry(-0.40946261032505227) q[0];
ry(0.4389813230545354) q[7];
cx q[0],q[7];
ry(-0.05642150713516833) q[0];
ry(-2.139389790596926) q[7];
cx q[0],q[7];
ry(-2.684246197667535) q[1];
ry(1.907768847847991) q[2];
cx q[1],q[2];
ry(-1.4783692996173057) q[1];
ry(0.09939370482760257) q[2];
cx q[1],q[2];
ry(-1.2911258900081073) q[1];
ry(-0.8371758961932253) q[3];
cx q[1],q[3];
ry(3.0491071592118053) q[1];
ry(2.688800521708768) q[3];
cx q[1],q[3];
ry(-0.29785186642897177) q[1];
ry(2.404197986825009) q[4];
cx q[1],q[4];
ry(-1.2139307436458973) q[1];
ry(1.9292956627315485) q[4];
cx q[1],q[4];
ry(0.40807347139491973) q[1];
ry(1.9695628166590724) q[5];
cx q[1],q[5];
ry(1.1427230932799985) q[1];
ry(2.102344319278646) q[5];
cx q[1],q[5];
ry(1.0789458104297056) q[1];
ry(0.3012659111776177) q[6];
cx q[1],q[6];
ry(-1.7552997330663016) q[1];
ry(2.2277615603798804) q[6];
cx q[1],q[6];
ry(2.7926358683011503) q[1];
ry(-0.016420648905024038) q[7];
cx q[1],q[7];
ry(0.08219163557279723) q[1];
ry(1.6600766592539913) q[7];
cx q[1],q[7];
ry(-2.2342239690200127) q[2];
ry(-2.279843534959137) q[3];
cx q[2],q[3];
ry(2.814299427322864) q[2];
ry(-2.749291570659526) q[3];
cx q[2],q[3];
ry(-1.838623392502873) q[2];
ry(-0.3335523986686484) q[4];
cx q[2],q[4];
ry(-1.4332355307229534) q[2];
ry(2.9800254663943395) q[4];
cx q[2],q[4];
ry(-0.1549045191413203) q[2];
ry(-0.5127169714272546) q[5];
cx q[2],q[5];
ry(-3.12383844899413) q[2];
ry(-1.6987985694883863) q[5];
cx q[2],q[5];
ry(1.9986207499161166) q[2];
ry(-2.137058552956183) q[6];
cx q[2],q[6];
ry(-2.119271195582903) q[2];
ry(3.0723951772694336) q[6];
cx q[2],q[6];
ry(-3.1113262979886946) q[2];
ry(-1.2173263900950932) q[7];
cx q[2],q[7];
ry(2.2051950532387194) q[2];
ry(-2.0993967178987654) q[7];
cx q[2],q[7];
ry(-2.894244013037835) q[3];
ry(-0.5294775842759307) q[4];
cx q[3],q[4];
ry(-3.063126934666163) q[3];
ry(-2.0238400755310835) q[4];
cx q[3],q[4];
ry(2.7532648250699694) q[3];
ry(-1.0010250441018451) q[5];
cx q[3],q[5];
ry(1.9356637488372082) q[3];
ry(1.8239964833678952) q[5];
cx q[3],q[5];
ry(-3.0691696919988756) q[3];
ry(-1.016259870301051) q[6];
cx q[3],q[6];
ry(-1.357264466701143) q[3];
ry(1.6990647966202277) q[6];
cx q[3],q[6];
ry(-2.1592117776805706) q[3];
ry(1.0225307630722218) q[7];
cx q[3],q[7];
ry(-0.5405474664992278) q[3];
ry(0.6545222450226933) q[7];
cx q[3],q[7];
ry(-0.31008814586049716) q[4];
ry(0.3530579780902991) q[5];
cx q[4],q[5];
ry(0.7001316333674892) q[4];
ry(-0.8433197516751195) q[5];
cx q[4],q[5];
ry(-2.8918487799869217) q[4];
ry(-1.0235022027865863) q[6];
cx q[4],q[6];
ry(0.525593076091269) q[4];
ry(3.0596109367835544) q[6];
cx q[4],q[6];
ry(-2.876438359468165) q[4];
ry(-0.5563105628603212) q[7];
cx q[4],q[7];
ry(-2.623666292223962) q[4];
ry(3.0671762924918005) q[7];
cx q[4],q[7];
ry(-1.9437784715863489) q[5];
ry(-1.7847509648670776) q[6];
cx q[5],q[6];
ry(1.7415427750297834) q[5];
ry(0.40508526792836363) q[6];
cx q[5],q[6];
ry(0.026829179886707966) q[5];
ry(2.2522642888843967) q[7];
cx q[5],q[7];
ry(-0.7363181428493522) q[5];
ry(-2.2938356972672405) q[7];
cx q[5],q[7];
ry(-0.9760118184862011) q[6];
ry(1.2604437337786045) q[7];
cx q[6],q[7];
ry(-0.39212416453496163) q[6];
ry(-0.7673049638304765) q[7];
cx q[6],q[7];
ry(3.124975974565548) q[0];
ry(-0.7787599690916152) q[1];
cx q[0],q[1];
ry(-0.8587524856708564) q[0];
ry(1.7380566290141433) q[1];
cx q[0],q[1];
ry(1.1483249349082456) q[0];
ry(-1.0938361145743405) q[2];
cx q[0],q[2];
ry(-2.145831598274018) q[0];
ry(2.190846042447135) q[2];
cx q[0],q[2];
ry(2.233203790752821) q[0];
ry(-0.9664762593059182) q[3];
cx q[0],q[3];
ry(2.223359090084346) q[0];
ry(2.043547815486548) q[3];
cx q[0],q[3];
ry(-1.6459162727317267) q[0];
ry(2.206550169406677) q[4];
cx q[0],q[4];
ry(-2.401252437194178) q[0];
ry(-2.004235664747566) q[4];
cx q[0],q[4];
ry(-0.7021363093104441) q[0];
ry(0.027598784231619853) q[5];
cx q[0],q[5];
ry(-0.027542384474714684) q[0];
ry(-1.3924113556846605) q[5];
cx q[0],q[5];
ry(2.901164951673754) q[0];
ry(0.7272878644953789) q[6];
cx q[0],q[6];
ry(0.35616327317447816) q[0];
ry(-0.2185672141887033) q[6];
cx q[0],q[6];
ry(0.06728722881132286) q[0];
ry(3.04281273655303) q[7];
cx q[0],q[7];
ry(-2.342429779375508) q[0];
ry(0.08952532220486553) q[7];
cx q[0],q[7];
ry(-1.1385840640077252) q[1];
ry(-0.5584263542123192) q[2];
cx q[1],q[2];
ry(2.7497437571998495) q[1];
ry(-2.362243631118751) q[2];
cx q[1],q[2];
ry(-1.7526692637061634) q[1];
ry(2.8198716709373097) q[3];
cx q[1],q[3];
ry(-2.639579060425215) q[1];
ry(2.551950537120725) q[3];
cx q[1],q[3];
ry(-2.8083403842835803) q[1];
ry(-2.550494031762227) q[4];
cx q[1],q[4];
ry(1.7513649077746027) q[1];
ry(3.0963753534368794) q[4];
cx q[1],q[4];
ry(2.052575235130392) q[1];
ry(-0.3605908459015168) q[5];
cx q[1],q[5];
ry(2.2818927206823023) q[1];
ry(-2.3275742503264736) q[5];
cx q[1],q[5];
ry(-1.2241476375206526) q[1];
ry(0.5877954690014366) q[6];
cx q[1],q[6];
ry(1.3514262272731905) q[1];
ry(-2.161165105363006) q[6];
cx q[1],q[6];
ry(-1.8465523164999054) q[1];
ry(1.7657731972570403) q[7];
cx q[1],q[7];
ry(-1.9826167779845747) q[1];
ry(-1.900145532578677) q[7];
cx q[1],q[7];
ry(-1.1751323246653882) q[2];
ry(-2.9220158417672817) q[3];
cx q[2],q[3];
ry(-0.07886760623034306) q[2];
ry(-0.23135528830828944) q[3];
cx q[2],q[3];
ry(-1.0696420555281054) q[2];
ry(1.6927077518647293) q[4];
cx q[2],q[4];
ry(-2.1401980571514025) q[2];
ry(2.451363972376906) q[4];
cx q[2],q[4];
ry(-0.10211464357441892) q[2];
ry(2.080515043710755) q[5];
cx q[2],q[5];
ry(-1.6610222901154545) q[2];
ry(-2.0994202064456218) q[5];
cx q[2],q[5];
ry(-2.4086701249463434) q[2];
ry(1.2844571795366793) q[6];
cx q[2],q[6];
ry(2.7096963969024372) q[2];
ry(-0.5615215230880395) q[6];
cx q[2],q[6];
ry(-1.4256641175757176) q[2];
ry(2.0689364573205196) q[7];
cx q[2],q[7];
ry(-0.12673908434033102) q[2];
ry(-1.974541964967204) q[7];
cx q[2],q[7];
ry(-1.8387473677409885) q[3];
ry(0.8351696170491403) q[4];
cx q[3],q[4];
ry(-2.896063948172519) q[3];
ry(-1.1296576566032024) q[4];
cx q[3],q[4];
ry(1.0311659470346655) q[3];
ry(0.0013660256371155462) q[5];
cx q[3],q[5];
ry(-1.0984662756054449) q[3];
ry(-0.5410509925274788) q[5];
cx q[3],q[5];
ry(-0.29049060088005024) q[3];
ry(1.1029208116275422) q[6];
cx q[3],q[6];
ry(2.8028422012819187) q[3];
ry(0.3993578402172018) q[6];
cx q[3],q[6];
ry(2.2605300683434857) q[3];
ry(-1.628272396872182) q[7];
cx q[3],q[7];
ry(1.7778624227441304) q[3];
ry(-0.17001564505844421) q[7];
cx q[3],q[7];
ry(2.828195662999861) q[4];
ry(0.8301948464032388) q[5];
cx q[4],q[5];
ry(2.555739060337458) q[4];
ry(-0.7009967591940685) q[5];
cx q[4],q[5];
ry(1.9308744184693565) q[4];
ry(1.0555381292132782) q[6];
cx q[4],q[6];
ry(-0.7369296618263158) q[4];
ry(0.32679984751502467) q[6];
cx q[4],q[6];
ry(-1.2005837115439544) q[4];
ry(1.1739705636891753) q[7];
cx q[4],q[7];
ry(1.938053699379135) q[4];
ry(-1.4740517585446482) q[7];
cx q[4],q[7];
ry(-0.42577999866109106) q[5];
ry(1.622694273943665) q[6];
cx q[5],q[6];
ry(2.247363013256421) q[5];
ry(2.5241136634783445) q[6];
cx q[5],q[6];
ry(-2.7705777607405215) q[5];
ry(2.163461573691837) q[7];
cx q[5],q[7];
ry(-0.1679065990739565) q[5];
ry(-1.2285875302556626) q[7];
cx q[5],q[7];
ry(-0.39701507416324233) q[6];
ry(-2.0489866924910167) q[7];
cx q[6],q[7];
ry(-0.4108754834192912) q[6];
ry(0.6809125502792588) q[7];
cx q[6],q[7];
ry(2.1349865332418174) q[0];
ry(2.188487821103087) q[1];
cx q[0],q[1];
ry(-2.6830045292713023) q[0];
ry(-2.8570429995587427) q[1];
cx q[0],q[1];
ry(-1.3947707833731018) q[0];
ry(2.6312012688170205) q[2];
cx q[0],q[2];
ry(1.8149083611522565) q[0];
ry(0.6524739495267707) q[2];
cx q[0],q[2];
ry(-1.127099148306284) q[0];
ry(-0.41956271640379134) q[3];
cx q[0],q[3];
ry(-1.0846671784165287) q[0];
ry(2.3474076960544097) q[3];
cx q[0],q[3];
ry(2.534962826796854) q[0];
ry(0.9164641231010968) q[4];
cx q[0],q[4];
ry(0.449973228447373) q[0];
ry(-0.287936244078298) q[4];
cx q[0],q[4];
ry(-0.05764450268474519) q[0];
ry(1.1539047713081452) q[5];
cx q[0],q[5];
ry(-0.1830279858128356) q[0];
ry(-2.962149051208653) q[5];
cx q[0],q[5];
ry(2.2732632417588876) q[0];
ry(0.5577445342865328) q[6];
cx q[0],q[6];
ry(-1.822138306011368) q[0];
ry(-2.367638220322003) q[6];
cx q[0],q[6];
ry(2.3177056480361817) q[0];
ry(-2.188641917195168) q[7];
cx q[0],q[7];
ry(3.1352484158032676) q[0];
ry(2.7624730051463087) q[7];
cx q[0],q[7];
ry(-1.1249538415898774) q[1];
ry(-2.1277747249272227) q[2];
cx q[1],q[2];
ry(-1.584146265131837) q[1];
ry(2.1707622140585965) q[2];
cx q[1],q[2];
ry(-0.8932688007036294) q[1];
ry(-1.7300634033664135) q[3];
cx q[1],q[3];
ry(-1.747715131587376) q[1];
ry(2.3101871516466588) q[3];
cx q[1],q[3];
ry(1.0205679952708895) q[1];
ry(0.5946418943073061) q[4];
cx q[1],q[4];
ry(-2.7423616483169972) q[1];
ry(3.113570840700664) q[4];
cx q[1],q[4];
ry(-0.824029756398799) q[1];
ry(-0.8296310025876129) q[5];
cx q[1],q[5];
ry(0.33833611401559954) q[1];
ry(-2.840816804365181) q[5];
cx q[1],q[5];
ry(-2.865790936121068) q[1];
ry(-1.900816606743625) q[6];
cx q[1],q[6];
ry(1.9122017283800232) q[1];
ry(1.265644526884829) q[6];
cx q[1],q[6];
ry(-2.2991722879595193) q[1];
ry(-1.695312545780505) q[7];
cx q[1],q[7];
ry(-0.18261938738570738) q[1];
ry(0.37055718483145134) q[7];
cx q[1],q[7];
ry(-0.019930959987155994) q[2];
ry(-1.7891090027310301) q[3];
cx q[2],q[3];
ry(-1.5363369037996737) q[2];
ry(2.443591969286515) q[3];
cx q[2],q[3];
ry(1.6177465430416502) q[2];
ry(-0.49350304806254197) q[4];
cx q[2],q[4];
ry(-2.597502329105151) q[2];
ry(2.2169482630372466) q[4];
cx q[2],q[4];
ry(0.9369477642009244) q[2];
ry(-2.151954350959496) q[5];
cx q[2],q[5];
ry(-2.8239791730464328) q[2];
ry(0.30225631789992213) q[5];
cx q[2],q[5];
ry(-1.0579178129206954) q[2];
ry(-1.0258692928148538) q[6];
cx q[2],q[6];
ry(-1.9099222336302075) q[2];
ry(1.4921573199581264) q[6];
cx q[2],q[6];
ry(-2.468697820655285) q[2];
ry(-0.14659183260625305) q[7];
cx q[2],q[7];
ry(-1.9015907112059107) q[2];
ry(0.5527518402572813) q[7];
cx q[2],q[7];
ry(2.459342956733936) q[3];
ry(3.026162639204015) q[4];
cx q[3],q[4];
ry(2.241106754993827) q[3];
ry(0.14131200228325455) q[4];
cx q[3],q[4];
ry(-2.6718639521097383) q[3];
ry(0.4672815347717849) q[5];
cx q[3],q[5];
ry(1.2654020682654927) q[3];
ry(-2.867883720833983) q[5];
cx q[3],q[5];
ry(1.8347623762785406) q[3];
ry(3.0135734196671318) q[6];
cx q[3],q[6];
ry(-1.5181976308491556) q[3];
ry(1.358564862719753) q[6];
cx q[3],q[6];
ry(-3.1260710897634536) q[3];
ry(-2.697476243952278) q[7];
cx q[3],q[7];
ry(-0.8954515958241576) q[3];
ry(-3.072514162466524) q[7];
cx q[3],q[7];
ry(1.591171100324087) q[4];
ry(0.9316886167140765) q[5];
cx q[4],q[5];
ry(1.6136369934334382) q[4];
ry(1.4672642870158112) q[5];
cx q[4],q[5];
ry(-0.3830444035424492) q[4];
ry(-2.8948521146939727) q[6];
cx q[4],q[6];
ry(-2.5184587643491216) q[4];
ry(-1.6536008360727175) q[6];
cx q[4],q[6];
ry(-0.3837307138970597) q[4];
ry(-0.8193968043495712) q[7];
cx q[4],q[7];
ry(-1.3942425526419253) q[4];
ry(0.16525891679782384) q[7];
cx q[4],q[7];
ry(1.7121001860863394) q[5];
ry(1.4165080061228599) q[6];
cx q[5],q[6];
ry(-1.2766884084886698) q[5];
ry(1.6176706829213026) q[6];
cx q[5],q[6];
ry(-0.9687201897788595) q[5];
ry(-2.9646334800203866) q[7];
cx q[5],q[7];
ry(-1.8576379932225728) q[5];
ry(-1.5601481639807258) q[7];
cx q[5],q[7];
ry(0.7231747584777418) q[6];
ry(3.067243059783166) q[7];
cx q[6],q[7];
ry(-2.010686024466284) q[6];
ry(0.9749111635601873) q[7];
cx q[6],q[7];
ry(-0.8516798929927214) q[0];
ry(-1.6288056797249328) q[1];
cx q[0],q[1];
ry(-1.641539947716088) q[0];
ry(-2.467057249951061) q[1];
cx q[0],q[1];
ry(-1.2879895661375533) q[0];
ry(0.782772058286867) q[2];
cx q[0],q[2];
ry(-0.6040611244441132) q[0];
ry(-0.033954982571904324) q[2];
cx q[0],q[2];
ry(-3.067008878053985) q[0];
ry(1.7773850619759435) q[3];
cx q[0],q[3];
ry(1.964404081397957) q[0];
ry(-0.1361405103373423) q[3];
cx q[0],q[3];
ry(-2.5242882725621625) q[0];
ry(-2.6964249302885492) q[4];
cx q[0],q[4];
ry(3.1009953307597997) q[0];
ry(-3.0300573263474773) q[4];
cx q[0],q[4];
ry(0.9995485668109163) q[0];
ry(1.725910868897518) q[5];
cx q[0],q[5];
ry(-0.7706048453480454) q[0];
ry(-1.739264045016446) q[5];
cx q[0],q[5];
ry(-1.937115874970459) q[0];
ry(0.14526119117414993) q[6];
cx q[0],q[6];
ry(-2.901954803044523) q[0];
ry(2.4742646134834807) q[6];
cx q[0],q[6];
ry(0.6861292074204667) q[0];
ry(-2.2290297034936932) q[7];
cx q[0],q[7];
ry(2.2099839837586472) q[0];
ry(-2.3553274478691897) q[7];
cx q[0],q[7];
ry(-2.528668225540622) q[1];
ry(1.3286718365995498) q[2];
cx q[1],q[2];
ry(0.5787353841411067) q[1];
ry(1.5409618315622449) q[2];
cx q[1],q[2];
ry(-2.651956395890039) q[1];
ry(2.817443731972096) q[3];
cx q[1],q[3];
ry(-2.879556013344391) q[1];
ry(2.3258783334538435) q[3];
cx q[1],q[3];
ry(0.24922923923777648) q[1];
ry(1.7113573922536185) q[4];
cx q[1],q[4];
ry(0.79857028499615) q[1];
ry(-1.7162477032274246) q[4];
cx q[1],q[4];
ry(0.48322796139145374) q[1];
ry(-1.678052645727762) q[5];
cx q[1],q[5];
ry(-0.871177695383624) q[1];
ry(-1.4319285652901357) q[5];
cx q[1],q[5];
ry(-3.034872141437816) q[1];
ry(-2.697480358103936) q[6];
cx q[1],q[6];
ry(2.715976149677671) q[1];
ry(0.34275440043106287) q[6];
cx q[1],q[6];
ry(3.1159099384037154) q[1];
ry(0.8394857656941452) q[7];
cx q[1],q[7];
ry(-2.680937542297403) q[1];
ry(-0.21875825482359) q[7];
cx q[1],q[7];
ry(-1.7234068861205583) q[2];
ry(-3.12130723775204) q[3];
cx q[2],q[3];
ry(2.709205827013621) q[2];
ry(-0.35777518885867426) q[3];
cx q[2],q[3];
ry(-2.087350925320066) q[2];
ry(-1.392304024007304) q[4];
cx q[2],q[4];
ry(1.0783210277245383) q[2];
ry(-2.9680860781013307) q[4];
cx q[2],q[4];
ry(1.9166810326627721) q[2];
ry(1.4604459680379667) q[5];
cx q[2],q[5];
ry(-2.4151188468759157) q[2];
ry(0.6176937083447216) q[5];
cx q[2],q[5];
ry(0.6645027106918142) q[2];
ry(1.70140985157526) q[6];
cx q[2],q[6];
ry(2.6361384156671415) q[2];
ry(-0.3238174291651741) q[6];
cx q[2],q[6];
ry(0.7506307901111634) q[2];
ry(-0.04133670083419135) q[7];
cx q[2],q[7];
ry(0.28719208904417837) q[2];
ry(2.173255130256437) q[7];
cx q[2],q[7];
ry(-1.9561650213205306) q[3];
ry(-1.6969797626827825) q[4];
cx q[3],q[4];
ry(0.10726303332388727) q[3];
ry(0.6864552988848915) q[4];
cx q[3],q[4];
ry(0.6292892803175506) q[3];
ry(2.1833247248441303) q[5];
cx q[3],q[5];
ry(2.7816770220963254) q[3];
ry(-0.36791559192737733) q[5];
cx q[3],q[5];
ry(1.0956028452033726) q[3];
ry(0.7347450314506) q[6];
cx q[3],q[6];
ry(-1.1619466982174367) q[3];
ry(-1.4895161431393102) q[6];
cx q[3],q[6];
ry(-2.5864723960173754) q[3];
ry(-3.0682037819874868) q[7];
cx q[3],q[7];
ry(3.0887006110745903) q[3];
ry(-2.022510922495928) q[7];
cx q[3],q[7];
ry(0.48671042108488827) q[4];
ry(0.5921077158856374) q[5];
cx q[4],q[5];
ry(0.08442797818992491) q[4];
ry(2.5428357638084673) q[5];
cx q[4],q[5];
ry(0.0070000934622893324) q[4];
ry(-1.1094571726990416) q[6];
cx q[4],q[6];
ry(2.341887706536958) q[4];
ry(-2.7561628431839353) q[6];
cx q[4],q[6];
ry(-2.772151192223576) q[4];
ry(2.3891944662475746) q[7];
cx q[4],q[7];
ry(2.206476485450966) q[4];
ry(0.49196700114105857) q[7];
cx q[4],q[7];
ry(1.6520293138619762) q[5];
ry(3.0218717124523726) q[6];
cx q[5],q[6];
ry(2.65961318131901) q[5];
ry(2.6675692373561666) q[6];
cx q[5],q[6];
ry(-2.368806709727305) q[5];
ry(-2.43403310658866) q[7];
cx q[5],q[7];
ry(1.2134221697758418) q[5];
ry(1.6071632432293654) q[7];
cx q[5],q[7];
ry(2.1871705529618457) q[6];
ry(-0.8457836442229515) q[7];
cx q[6],q[7];
ry(-2.777135231826162) q[6];
ry(-2.2166123833871385) q[7];
cx q[6],q[7];
ry(1.2615161151251852) q[0];
ry(-0.4517742771138282) q[1];
cx q[0],q[1];
ry(-1.405369998822609) q[0];
ry(-0.18733045720155328) q[1];
cx q[0],q[1];
ry(2.1460131913911216) q[0];
ry(2.095191383943332) q[2];
cx q[0],q[2];
ry(2.781767282575363) q[0];
ry(2.1355054160040536) q[2];
cx q[0],q[2];
ry(-3.092735263860341) q[0];
ry(1.9596988063937482) q[3];
cx q[0],q[3];
ry(2.1227253625428624) q[0];
ry(2.30567360972485) q[3];
cx q[0],q[3];
ry(-2.7314116350709425) q[0];
ry(2.5415269663188496) q[4];
cx q[0],q[4];
ry(0.5459148739502595) q[0];
ry(-2.452426493328154) q[4];
cx q[0],q[4];
ry(3.0630624481187376) q[0];
ry(0.7931408607052086) q[5];
cx q[0],q[5];
ry(0.5821277975836541) q[0];
ry(-1.746870313085882) q[5];
cx q[0],q[5];
ry(-2.5177556923335076) q[0];
ry(-1.873024766146815) q[6];
cx q[0],q[6];
ry(-3.0713419772940704) q[0];
ry(-1.3769670657653976) q[6];
cx q[0],q[6];
ry(0.07939587189930819) q[0];
ry(0.5624841120625366) q[7];
cx q[0],q[7];
ry(-2.5860915944885767) q[0];
ry(1.1560811208498514) q[7];
cx q[0],q[7];
ry(1.1323669733539372) q[1];
ry(-0.014631301413624342) q[2];
cx q[1],q[2];
ry(2.9441121755264232) q[1];
ry(2.8021243985162747) q[2];
cx q[1],q[2];
ry(2.249731442623213) q[1];
ry(0.647300740381994) q[3];
cx q[1],q[3];
ry(-0.33923629091046337) q[1];
ry(-1.1265110375795202) q[3];
cx q[1],q[3];
ry(-2.8098149049371806) q[1];
ry(2.2022992426573413) q[4];
cx q[1],q[4];
ry(0.00784076486672821) q[1];
ry(-2.374824728826717) q[4];
cx q[1],q[4];
ry(1.7950334885887407) q[1];
ry(-2.7475782286594774) q[5];
cx q[1],q[5];
ry(1.597060304351926) q[1];
ry(-2.545488100077331) q[5];
cx q[1],q[5];
ry(2.9144285226247626) q[1];
ry(-0.319775037696175) q[6];
cx q[1],q[6];
ry(-0.2607972882805454) q[1];
ry(3.005965756354964) q[6];
cx q[1],q[6];
ry(1.9266990996163973) q[1];
ry(0.6086244136524872) q[7];
cx q[1],q[7];
ry(-0.8584774546475824) q[1];
ry(1.9249233733705813) q[7];
cx q[1],q[7];
ry(-0.5131871235041023) q[2];
ry(1.770704405247458) q[3];
cx q[2],q[3];
ry(1.1063480064120734) q[2];
ry(-2.4862626988942957) q[3];
cx q[2],q[3];
ry(-0.1008652297585213) q[2];
ry(-1.585776313268238) q[4];
cx q[2],q[4];
ry(-2.83166897269782) q[2];
ry(3.093265514196166) q[4];
cx q[2],q[4];
ry(-1.4182662231051788) q[2];
ry(-0.3926020388654062) q[5];
cx q[2],q[5];
ry(2.8863254115201236) q[2];
ry(-0.6768591477080187) q[5];
cx q[2],q[5];
ry(-0.16261287507050426) q[2];
ry(2.4119815345135187) q[6];
cx q[2],q[6];
ry(0.29009682217480925) q[2];
ry(0.7245582588124189) q[6];
cx q[2],q[6];
ry(-0.449730986348575) q[2];
ry(-0.872193924649581) q[7];
cx q[2],q[7];
ry(-1.15916244077092) q[2];
ry(0.03837462775021187) q[7];
cx q[2],q[7];
ry(1.348679771550007) q[3];
ry(0.5848370148334949) q[4];
cx q[3],q[4];
ry(0.09574310835198753) q[3];
ry(-0.7063560200188139) q[4];
cx q[3],q[4];
ry(0.514685439630485) q[3];
ry(0.7229754673980988) q[5];
cx q[3],q[5];
ry(-2.8590660187766535) q[3];
ry(1.0338626156883333) q[5];
cx q[3],q[5];
ry(-2.6339960454099995) q[3];
ry(1.9350373417375566) q[6];
cx q[3],q[6];
ry(0.1624225191344563) q[3];
ry(0.8500855132121569) q[6];
cx q[3],q[6];
ry(-1.9302083690666585) q[3];
ry(1.8877788023759037) q[7];
cx q[3],q[7];
ry(0.10865198041079172) q[3];
ry(2.0450401365666395) q[7];
cx q[3],q[7];
ry(-2.3158660755223086) q[4];
ry(-0.13198970756612258) q[5];
cx q[4],q[5];
ry(1.0602901531052087) q[4];
ry(0.09543693307730165) q[5];
cx q[4],q[5];
ry(2.10393957751546) q[4];
ry(-1.408806203708775) q[6];
cx q[4],q[6];
ry(-3.1259223981262774) q[4];
ry(1.4787259182444756) q[6];
cx q[4],q[6];
ry(-2.6565364530625435) q[4];
ry(1.7448742112807227) q[7];
cx q[4],q[7];
ry(1.17925849311976) q[4];
ry(-2.286444038267169) q[7];
cx q[4],q[7];
ry(1.6244651936076675) q[5];
ry(2.7224683896005843) q[6];
cx q[5],q[6];
ry(-0.6933013647425597) q[5];
ry(-0.08471997871937731) q[6];
cx q[5],q[6];
ry(0.2969520910534976) q[5];
ry(2.2777723291415275) q[7];
cx q[5],q[7];
ry(3.015443830335543) q[5];
ry(-1.6840379708975453) q[7];
cx q[5],q[7];
ry(2.699943521441508) q[6];
ry(-2.9694601843019655) q[7];
cx q[6],q[7];
ry(2.843248440160661) q[6];
ry(1.5771681048279078) q[7];
cx q[6],q[7];
ry(2.809575202865781) q[0];
ry(2.0586124320658428) q[1];
cx q[0],q[1];
ry(-1.740445001137559) q[0];
ry(-2.69170185454805) q[1];
cx q[0],q[1];
ry(-0.7030709114987868) q[0];
ry(-2.636823247191302) q[2];
cx q[0],q[2];
ry(-2.2244475886050714) q[0];
ry(-2.5617720662451253) q[2];
cx q[0],q[2];
ry(2.1399459840962303) q[0];
ry(2.182842146897971) q[3];
cx q[0],q[3];
ry(-2.3780160707649545) q[0];
ry(-2.4671397331736857) q[3];
cx q[0],q[3];
ry(0.02580589742866035) q[0];
ry(-0.7873994685582248) q[4];
cx q[0],q[4];
ry(-1.81091707124592) q[0];
ry(1.967037855076253) q[4];
cx q[0],q[4];
ry(2.5200435820827596) q[0];
ry(-1.6743207736480281) q[5];
cx q[0],q[5];
ry(-2.3880659721023125) q[0];
ry(-0.0013135736134351579) q[5];
cx q[0],q[5];
ry(3.026177024658372) q[0];
ry(-0.23767816197329594) q[6];
cx q[0],q[6];
ry(0.9294852201272361) q[0];
ry(0.07068287055115974) q[6];
cx q[0],q[6];
ry(1.693326395591735) q[0];
ry(1.0749131848108597) q[7];
cx q[0],q[7];
ry(2.474473576667622) q[0];
ry(-1.3206875244579759) q[7];
cx q[0],q[7];
ry(-1.0027930576752695) q[1];
ry(-0.6663805747562286) q[2];
cx q[1],q[2];
ry(1.7142498880791135) q[1];
ry(-2.4649810426489003) q[2];
cx q[1],q[2];
ry(0.9410539985770158) q[1];
ry(0.12785440092030598) q[3];
cx q[1],q[3];
ry(-2.0779898652966913) q[1];
ry(-1.077220260340833) q[3];
cx q[1],q[3];
ry(1.1454276664385281) q[1];
ry(-0.48555887640975826) q[4];
cx q[1],q[4];
ry(2.7820928847873723) q[1];
ry(-2.6365033566000364) q[4];
cx q[1],q[4];
ry(0.41712008074950097) q[1];
ry(2.377273821289923) q[5];
cx q[1],q[5];
ry(-1.162676433391165) q[1];
ry(-2.7132713172629104) q[5];
cx q[1],q[5];
ry(0.6932621604404019) q[1];
ry(-1.1300598370951631) q[6];
cx q[1],q[6];
ry(-0.9860188069182972) q[1];
ry(-2.6326378336590563) q[6];
cx q[1],q[6];
ry(-1.469807402646153) q[1];
ry(1.0643211326249735) q[7];
cx q[1],q[7];
ry(-2.1104504322089532) q[1];
ry(-1.872411997384505) q[7];
cx q[1],q[7];
ry(-1.9502610108096015) q[2];
ry(2.7733760123665228) q[3];
cx q[2],q[3];
ry(-2.143058622443125) q[2];
ry(-2.6196447943786536) q[3];
cx q[2],q[3];
ry(-2.612132556859556) q[2];
ry(-0.63012226202002) q[4];
cx q[2],q[4];
ry(0.5195587337256664) q[2];
ry(-2.4882085055409884) q[4];
cx q[2],q[4];
ry(-1.4508080443499525) q[2];
ry(-2.9824126243923814) q[5];
cx q[2],q[5];
ry(-0.44636726504313984) q[2];
ry(-1.9527633016352222) q[5];
cx q[2],q[5];
ry(-2.1694995509722332) q[2];
ry(0.6459050279258274) q[6];
cx q[2],q[6];
ry(-2.001374356211117) q[2];
ry(1.9301965732534465) q[6];
cx q[2],q[6];
ry(-2.6589188165960067) q[2];
ry(-0.9640593286496851) q[7];
cx q[2],q[7];
ry(-0.4705673464386535) q[2];
ry(3.1299226799960453) q[7];
cx q[2],q[7];
ry(-0.7385490696331333) q[3];
ry(1.1961617403796918) q[4];
cx q[3],q[4];
ry(1.964667503454208) q[3];
ry(0.8811396105034737) q[4];
cx q[3],q[4];
ry(1.7766524034731326) q[3];
ry(0.07780929444594253) q[5];
cx q[3],q[5];
ry(-1.6933065033224501) q[3];
ry(1.5660906549968021) q[5];
cx q[3],q[5];
ry(-0.12170330783335705) q[3];
ry(0.6432994510066007) q[6];
cx q[3],q[6];
ry(1.6601044153144322) q[3];
ry(0.5088575823503776) q[6];
cx q[3],q[6];
ry(0.04197936125635593) q[3];
ry(-2.0837060577367525) q[7];
cx q[3],q[7];
ry(-1.1731078144589597) q[3];
ry(-0.555857153694192) q[7];
cx q[3],q[7];
ry(-2.8200757563708754) q[4];
ry(0.0034762044487415267) q[5];
cx q[4],q[5];
ry(0.7779439373731303) q[4];
ry(0.8084854131673467) q[5];
cx q[4],q[5];
ry(-2.007421837729184) q[4];
ry(2.048626639718163) q[6];
cx q[4],q[6];
ry(-0.051476257909094984) q[4];
ry(-0.4927032479238971) q[6];
cx q[4],q[6];
ry(0.1468969389493564) q[4];
ry(2.0431962718019596) q[7];
cx q[4],q[7];
ry(2.820413806785495) q[4];
ry(1.997467635284301) q[7];
cx q[4],q[7];
ry(-0.320216491730406) q[5];
ry(1.0156926933652466) q[6];
cx q[5],q[6];
ry(-1.4251208173640535) q[5];
ry(-1.5298646946249495) q[6];
cx q[5],q[6];
ry(-1.8696702359264377) q[5];
ry(-1.5948395814867533) q[7];
cx q[5],q[7];
ry(2.0189335059335836) q[5];
ry(3.046641437041777) q[7];
cx q[5],q[7];
ry(0.9270742926042838) q[6];
ry(-1.244112651412908) q[7];
cx q[6],q[7];
ry(1.1396255314293335) q[6];
ry(0.6735132181666064) q[7];
cx q[6],q[7];
ry(1.6613785411290543) q[0];
ry(-0.019089143488129363) q[1];
cx q[0],q[1];
ry(-0.02408886326903925) q[0];
ry(1.2691595062264627) q[1];
cx q[0],q[1];
ry(-0.43490263282994235) q[0];
ry(1.17302646641259) q[2];
cx q[0],q[2];
ry(-2.8618932747867283) q[0];
ry(-3.056209578849746) q[2];
cx q[0],q[2];
ry(1.4865733456054286) q[0];
ry(2.7042350814449287) q[3];
cx q[0],q[3];
ry(-0.45400024837056385) q[0];
ry(-2.2602266369865216) q[3];
cx q[0],q[3];
ry(-2.6721215571388113) q[0];
ry(-0.023858192857807346) q[4];
cx q[0],q[4];
ry(-0.21428906204601234) q[0];
ry(2.8822445582630953) q[4];
cx q[0],q[4];
ry(1.8500343696597135) q[0];
ry(-2.8667053202177515) q[5];
cx q[0],q[5];
ry(2.0587960674071484) q[0];
ry(2.1863183696538195) q[5];
cx q[0],q[5];
ry(2.696152156159465) q[0];
ry(-0.059989888282823145) q[6];
cx q[0],q[6];
ry(-0.9168442593906638) q[0];
ry(2.3504987380228064) q[6];
cx q[0],q[6];
ry(-0.1376471518340514) q[0];
ry(3.1000430617891985) q[7];
cx q[0],q[7];
ry(-2.490240879385276) q[0];
ry(1.501527218705909) q[7];
cx q[0],q[7];
ry(1.0525011159555901) q[1];
ry(2.5836053122443157) q[2];
cx q[1],q[2];
ry(0.670176112247221) q[1];
ry(-2.7402053081449123) q[2];
cx q[1],q[2];
ry(0.7220470569755912) q[1];
ry(2.646439696343258) q[3];
cx q[1],q[3];
ry(0.2875366564225529) q[1];
ry(-3.0948151114549542) q[3];
cx q[1],q[3];
ry(-1.285259118805829) q[1];
ry(-0.5660691133328539) q[4];
cx q[1],q[4];
ry(-2.5259279646404083) q[1];
ry(2.1604085911226) q[4];
cx q[1],q[4];
ry(0.6031113495395255) q[1];
ry(1.2652300674574177) q[5];
cx q[1],q[5];
ry(-2.9876071732698324) q[1];
ry(0.0006881892588710415) q[5];
cx q[1],q[5];
ry(-2.9961943433825287) q[1];
ry(-2.997890445784365) q[6];
cx q[1],q[6];
ry(-0.49319192207702195) q[1];
ry(-1.7904706840824034) q[6];
cx q[1],q[6];
ry(-2.3142708074632496) q[1];
ry(-2.3144243299252776) q[7];
cx q[1],q[7];
ry(-1.1116318466433563) q[1];
ry(1.759864810091603) q[7];
cx q[1],q[7];
ry(-0.5059957252479879) q[2];
ry(-2.2126592431959544) q[3];
cx q[2],q[3];
ry(2.7837857818520435) q[2];
ry(0.5561292376760684) q[3];
cx q[2],q[3];
ry(2.4127318509403333) q[2];
ry(0.04265555775496477) q[4];
cx q[2],q[4];
ry(0.3276908306431761) q[2];
ry(0.9705642438343318) q[4];
cx q[2],q[4];
ry(-0.4645453777601349) q[2];
ry(-2.079220347384958) q[5];
cx q[2],q[5];
ry(2.4148453612696468) q[2];
ry(0.3747932887458704) q[5];
cx q[2],q[5];
ry(-0.43800011536371264) q[2];
ry(-1.5525932606452049) q[6];
cx q[2],q[6];
ry(-2.4098738848533077) q[2];
ry(1.5014533149145992) q[6];
cx q[2],q[6];
ry(0.39353094154495794) q[2];
ry(-2.203133198334587) q[7];
cx q[2],q[7];
ry(0.6287959815986106) q[2];
ry(2.478095248331712) q[7];
cx q[2],q[7];
ry(-2.3401773038279936) q[3];
ry(1.8878115457138227) q[4];
cx q[3],q[4];
ry(0.05690462652741951) q[3];
ry(2.3402945904487664) q[4];
cx q[3],q[4];
ry(2.9025648018482695) q[3];
ry(2.44383488927003) q[5];
cx q[3],q[5];
ry(-1.8612260676073573) q[3];
ry(0.5251860417216185) q[5];
cx q[3],q[5];
ry(-1.0367747934057012) q[3];
ry(-0.3414818910177706) q[6];
cx q[3],q[6];
ry(-1.6280377904212204) q[3];
ry(0.620912360935101) q[6];
cx q[3],q[6];
ry(-3.0016498685871) q[3];
ry(2.392903495707132) q[7];
cx q[3],q[7];
ry(-2.324545193518058) q[3];
ry(2.403288312396941) q[7];
cx q[3],q[7];
ry(-1.3920924584965135) q[4];
ry(0.19400381257102947) q[5];
cx q[4],q[5];
ry(0.779306912296705) q[4];
ry(0.05652434133687123) q[5];
cx q[4],q[5];
ry(1.988217184529922) q[4];
ry(-0.3817033543231354) q[6];
cx q[4],q[6];
ry(2.6963099937956394) q[4];
ry(-1.0092133171445328) q[6];
cx q[4],q[6];
ry(2.3720971524790917) q[4];
ry(-3.096203175765959) q[7];
cx q[4],q[7];
ry(-2.0547375338674048) q[4];
ry(0.23546079708380696) q[7];
cx q[4],q[7];
ry(2.8555107498415278) q[5];
ry(-2.2066866180148925) q[6];
cx q[5],q[6];
ry(2.4690659813299236) q[5];
ry(-1.550637058772863) q[6];
cx q[5],q[6];
ry(-2.5222000229763792) q[5];
ry(-2.8315869757132766) q[7];
cx q[5],q[7];
ry(0.12022358790711252) q[5];
ry(-1.324271576956237) q[7];
cx q[5],q[7];
ry(-1.9310320994387897) q[6];
ry(-1.985168909465905) q[7];
cx q[6],q[7];
ry(-2.644357918837061) q[6];
ry(2.091282262478023) q[7];
cx q[6],q[7];
ry(-2.065297615855554) q[0];
ry(-1.9350561556824974) q[1];
ry(-2.5376848673821772) q[2];
ry(0.3576628952753987) q[3];
ry(2.6425172455804327) q[4];
ry(-0.5652398117704749) q[5];
ry(-3.0075157083049278) q[6];
ry(2.2786104419288638) q[7];