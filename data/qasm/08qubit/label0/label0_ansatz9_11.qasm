OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.2981645701974278) q[0];
ry(2.8545535160107085) q[1];
cx q[0],q[1];
ry(-2.084500843410283) q[0];
ry(2.768638560029603) q[1];
cx q[0],q[1];
ry(-2.8443747756290745) q[2];
ry(1.3960578207614418) q[3];
cx q[2],q[3];
ry(0.7048955905250278) q[2];
ry(0.7990057913044792) q[3];
cx q[2],q[3];
ry(-2.298658774084206) q[4];
ry(0.40484199827385) q[5];
cx q[4],q[5];
ry(2.768191338870442) q[4];
ry(2.6128583665603653) q[5];
cx q[4],q[5];
ry(0.6988631172308586) q[6];
ry(0.3955692053551303) q[7];
cx q[6],q[7];
ry(2.6764494844145386) q[6];
ry(-1.0889069818879054) q[7];
cx q[6],q[7];
ry(-0.35906928711783337) q[0];
ry(-1.2220808652145831) q[2];
cx q[0],q[2];
ry(2.4584265992016787) q[0];
ry(2.690955579499596) q[2];
cx q[0],q[2];
ry(0.09963221694746771) q[2];
ry(-2.3677546225357906) q[4];
cx q[2],q[4];
ry(-0.6368521955023282) q[2];
ry(1.782063700385342) q[4];
cx q[2],q[4];
ry(1.3335470468063715) q[4];
ry(3.0275555572471595) q[6];
cx q[4],q[6];
ry(-0.9317372380383054) q[4];
ry(-2.2216717094602356) q[6];
cx q[4],q[6];
ry(-0.8069640626451005) q[1];
ry(3.125796970473132) q[3];
cx q[1],q[3];
ry(2.9628814966578667) q[1];
ry(-1.7467050085238434) q[3];
cx q[1],q[3];
ry(3.005524573497558) q[3];
ry(0.5093822962905534) q[5];
cx q[3],q[5];
ry(1.900014525026938) q[3];
ry(1.0482737172221872) q[5];
cx q[3],q[5];
ry(-1.8862394106842446) q[5];
ry(2.976458589756118) q[7];
cx q[5],q[7];
ry(3.094252530153486) q[5];
ry(-1.6720452305677531) q[7];
cx q[5],q[7];
ry(-2.65088144654981) q[0];
ry(-2.812850530961171) q[3];
cx q[0],q[3];
ry(-0.03859266735153799) q[0];
ry(-2.791543054426084) q[3];
cx q[0],q[3];
ry(-0.7349954120499812) q[1];
ry(-0.22045994113504938) q[2];
cx q[1],q[2];
ry(0.8270021949591483) q[1];
ry(-0.9018580568336771) q[2];
cx q[1],q[2];
ry(2.6069589608721624) q[2];
ry(0.22428302421579538) q[5];
cx q[2],q[5];
ry(0.6721855558416845) q[2];
ry(-2.541399589333962) q[5];
cx q[2],q[5];
ry(-1.745781029488363) q[3];
ry(-1.5611500944481085) q[4];
cx q[3],q[4];
ry(-0.45130008526542104) q[3];
ry(-1.7743749774318909) q[4];
cx q[3],q[4];
ry(2.693841151345044) q[4];
ry(0.2353344037380891) q[7];
cx q[4],q[7];
ry(-2.8242110267681944) q[4];
ry(0.5824174198509133) q[7];
cx q[4],q[7];
ry(-1.023305641557057) q[5];
ry(-0.14968376215054371) q[6];
cx q[5],q[6];
ry(0.29741965967580697) q[5];
ry(1.5806491994202) q[6];
cx q[5],q[6];
ry(1.8875914278858357) q[0];
ry(0.22167697974087733) q[1];
cx q[0],q[1];
ry(1.908354847143432) q[0];
ry(-0.3237766236609927) q[1];
cx q[0],q[1];
ry(-0.8914517618806022) q[2];
ry(0.6191320596339738) q[3];
cx q[2],q[3];
ry(-1.8456075807253915) q[2];
ry(1.9625474873810091) q[3];
cx q[2],q[3];
ry(-0.7633789792588725) q[4];
ry(-0.21281032381816828) q[5];
cx q[4],q[5];
ry(0.42035407679299563) q[4];
ry(-1.6125377497317182) q[5];
cx q[4],q[5];
ry(0.4849552189965141) q[6];
ry(-2.87872174823596) q[7];
cx q[6],q[7];
ry(3.12325429921102) q[6];
ry(-0.8134891993412351) q[7];
cx q[6],q[7];
ry(1.2457565337191208) q[0];
ry(1.898462223793951) q[2];
cx q[0],q[2];
ry(-2.5321292089810488) q[0];
ry(0.7522643505973576) q[2];
cx q[0],q[2];
ry(1.305886036703258) q[2];
ry(-0.6392938992813209) q[4];
cx q[2],q[4];
ry(-2.3170089330062673) q[2];
ry(-0.9283538931716491) q[4];
cx q[2],q[4];
ry(-1.318081995471685) q[4];
ry(-0.7187290886225876) q[6];
cx q[4],q[6];
ry(-1.7254049144557166) q[4];
ry(0.9997447875819702) q[6];
cx q[4],q[6];
ry(-2.7757673278270856) q[1];
ry(-1.1521944707967418) q[3];
cx q[1],q[3];
ry(-1.163872805509583) q[1];
ry(2.1478239050449206) q[3];
cx q[1],q[3];
ry(-0.593422400544043) q[3];
ry(-1.8732283752649497) q[5];
cx q[3],q[5];
ry(0.8966624164745447) q[3];
ry(1.6498063384521011) q[5];
cx q[3],q[5];
ry(1.8256436296191234) q[5];
ry(-1.6310865861547938) q[7];
cx q[5],q[7];
ry(0.8167490768674717) q[5];
ry(0.22838698735636975) q[7];
cx q[5],q[7];
ry(-2.406275812510599) q[0];
ry(-0.04490669125120209) q[3];
cx q[0],q[3];
ry(-0.05347026892379833) q[0];
ry(-0.39506324905672496) q[3];
cx q[0],q[3];
ry(0.6903617465533745) q[1];
ry(-2.4545347720234267) q[2];
cx q[1],q[2];
ry(0.0652782641452721) q[1];
ry(-1.6344733956555064) q[2];
cx q[1],q[2];
ry(1.7716840620413603) q[2];
ry(0.2902409780808508) q[5];
cx q[2],q[5];
ry(-1.5279975158735717) q[2];
ry(-0.26742280324425405) q[5];
cx q[2],q[5];
ry(-2.6614092820487216) q[3];
ry(-0.6435598365845511) q[4];
cx q[3],q[4];
ry(1.7731535273027603) q[3];
ry(-2.9395849038104838) q[4];
cx q[3],q[4];
ry(3.03077900793733) q[4];
ry(-0.8254183544653263) q[7];
cx q[4],q[7];
ry(-0.3691657982016041) q[4];
ry(-1.2463706179185925) q[7];
cx q[4],q[7];
ry(2.1998516521704046) q[5];
ry(0.3251812500679617) q[6];
cx q[5],q[6];
ry(-2.3032949028285437) q[5];
ry(-0.30364171125098455) q[6];
cx q[5],q[6];
ry(1.097288563865697) q[0];
ry(0.4526776024629455) q[1];
cx q[0],q[1];
ry(-0.348107389922878) q[0];
ry(2.716234062334148) q[1];
cx q[0],q[1];
ry(-1.3343201051619733) q[2];
ry(0.28626692270268383) q[3];
cx q[2],q[3];
ry(0.6231459470092339) q[2];
ry(2.125605817971345) q[3];
cx q[2],q[3];
ry(2.0116290732996736) q[4];
ry(-0.7332925264653946) q[5];
cx q[4],q[5];
ry(0.14090075934337776) q[4];
ry(-1.7201707117616558) q[5];
cx q[4],q[5];
ry(2.9562198266570534) q[6];
ry(-2.7244946751484065) q[7];
cx q[6],q[7];
ry(-2.2111243566872183) q[6];
ry(1.0591037499987142) q[7];
cx q[6],q[7];
ry(-0.9732141340869376) q[0];
ry(0.2580199809386725) q[2];
cx q[0],q[2];
ry(-2.894399311640583) q[0];
ry(1.0876415195385671) q[2];
cx q[0],q[2];
ry(0.5962316692136413) q[2];
ry(1.2938581637024582) q[4];
cx q[2],q[4];
ry(-0.34865606038146374) q[2];
ry(-0.41110592352925845) q[4];
cx q[2],q[4];
ry(-0.2865333997027967) q[4];
ry(-2.1989308659376205) q[6];
cx q[4],q[6];
ry(-1.7667235701592217) q[4];
ry(-2.8538036464215786) q[6];
cx q[4],q[6];
ry(-1.9317822722988307) q[1];
ry(2.457821776606782) q[3];
cx q[1],q[3];
ry(0.5706586242913056) q[1];
ry(0.9425140067784642) q[3];
cx q[1],q[3];
ry(0.9637185344544362) q[3];
ry(-0.2587282551432999) q[5];
cx q[3],q[5];
ry(2.7624959886531437) q[3];
ry(-2.4622403666949593) q[5];
cx q[3],q[5];
ry(0.33281545818098374) q[5];
ry(-2.3348644175484314) q[7];
cx q[5],q[7];
ry(0.04255670322935372) q[5];
ry(-0.22012321852263206) q[7];
cx q[5],q[7];
ry(-1.9544304338007334) q[0];
ry(-2.1175851649603574) q[3];
cx q[0],q[3];
ry(2.6127292284962453) q[0];
ry(0.48571034609394637) q[3];
cx q[0],q[3];
ry(0.19694455782262743) q[1];
ry(0.06413010028271061) q[2];
cx q[1],q[2];
ry(-1.6813299549719556) q[1];
ry(1.8180062370014727) q[2];
cx q[1],q[2];
ry(-0.8766949182243152) q[2];
ry(0.9715385513742558) q[5];
cx q[2],q[5];
ry(-0.2440788986812562) q[2];
ry(-2.1129282443843422) q[5];
cx q[2],q[5];
ry(1.0299076091621082) q[3];
ry(2.0143785509408354) q[4];
cx q[3],q[4];
ry(-0.260815684014184) q[3];
ry(3.0624333155790855) q[4];
cx q[3],q[4];
ry(2.2734293881654835) q[4];
ry(2.419548072695079) q[7];
cx q[4],q[7];
ry(1.9978101879721297) q[4];
ry(1.754577483493091) q[7];
cx q[4],q[7];
ry(1.425303211561714) q[5];
ry(-0.8895024747663675) q[6];
cx q[5],q[6];
ry(2.480436151849062) q[5];
ry(-0.5083185888258962) q[6];
cx q[5],q[6];
ry(-0.22028546930573992) q[0];
ry(2.349410827173259) q[1];
cx q[0],q[1];
ry(-2.6161417132427) q[0];
ry(2.7491570542066914) q[1];
cx q[0],q[1];
ry(0.7250427661807181) q[2];
ry(0.7222821970649784) q[3];
cx q[2],q[3];
ry(0.7156332199032605) q[2];
ry(1.4424398058322625) q[3];
cx q[2],q[3];
ry(-2.4295187134056437) q[4];
ry(1.0334097930464727) q[5];
cx q[4],q[5];
ry(-0.5266966438204079) q[4];
ry(2.9192196417548852) q[5];
cx q[4],q[5];
ry(-1.1792144346408548) q[6];
ry(-0.8626776028028758) q[7];
cx q[6],q[7];
ry(-0.632717313623) q[6];
ry(1.9955462255218164) q[7];
cx q[6],q[7];
ry(2.0858983687068227) q[0];
ry(-2.506231562254633) q[2];
cx q[0],q[2];
ry(2.1921772160816735) q[0];
ry(2.309001215774653) q[2];
cx q[0],q[2];
ry(1.0639602908200603) q[2];
ry(-0.8320195539036908) q[4];
cx q[2],q[4];
ry(-2.7539369921766927) q[2];
ry(2.4398670793361705) q[4];
cx q[2],q[4];
ry(2.505293103680413) q[4];
ry(-1.2637198363059023) q[6];
cx q[4],q[6];
ry(0.29062354746653263) q[4];
ry(2.475616682576002) q[6];
cx q[4],q[6];
ry(-2.1152199042747966) q[1];
ry(-0.4768812847275897) q[3];
cx q[1],q[3];
ry(-1.4202021837532073) q[1];
ry(0.9090833101406259) q[3];
cx q[1],q[3];
ry(1.1828658606687914) q[3];
ry(2.7121208037855205) q[5];
cx q[3],q[5];
ry(2.2669249728208483) q[3];
ry(2.1798986377342633) q[5];
cx q[3],q[5];
ry(1.8734337353811226) q[5];
ry(-2.617375673577052) q[7];
cx q[5],q[7];
ry(2.7714600267661065) q[5];
ry(2.2672942624714576) q[7];
cx q[5],q[7];
ry(-1.8822425997588847) q[0];
ry(0.3643854735501657) q[3];
cx q[0],q[3];
ry(-2.167458067445673) q[0];
ry(-0.34520566263643016) q[3];
cx q[0],q[3];
ry(-2.553210403351146) q[1];
ry(-2.2559040858238575) q[2];
cx q[1],q[2];
ry(1.404836520943836) q[1];
ry(-2.828063967432292) q[2];
cx q[1],q[2];
ry(2.6526813516625714) q[2];
ry(-0.35814054885154784) q[5];
cx q[2],q[5];
ry(-0.36679176456635754) q[2];
ry(-2.4077673356443876) q[5];
cx q[2],q[5];
ry(1.0136918313492043) q[3];
ry(1.6464147716640922) q[4];
cx q[3],q[4];
ry(-1.935081970356423) q[3];
ry(-0.008749146868467464) q[4];
cx q[3],q[4];
ry(-3.056565367183911) q[4];
ry(0.24986923085211907) q[7];
cx q[4],q[7];
ry(0.47505632791413116) q[4];
ry(1.0009449025749941) q[7];
cx q[4],q[7];
ry(3.0393404032764613) q[5];
ry(-2.9531668884843003) q[6];
cx q[5],q[6];
ry(-2.9276635880624373) q[5];
ry(-1.313436872815032) q[6];
cx q[5],q[6];
ry(0.8294238241560992) q[0];
ry(2.8137282819803624) q[1];
cx q[0],q[1];
ry(1.961281100764927) q[0];
ry(0.7379462426252941) q[1];
cx q[0],q[1];
ry(-1.418615340669831) q[2];
ry(2.89618349446229) q[3];
cx q[2],q[3];
ry(1.45698363498103) q[2];
ry(2.829623050492003) q[3];
cx q[2],q[3];
ry(-2.410569760914682) q[4];
ry(1.21650142229093) q[5];
cx q[4],q[5];
ry(-0.09009200719346738) q[4];
ry(-0.18889902894370686) q[5];
cx q[4],q[5];
ry(2.2334755635724033) q[6];
ry(0.37840248065824283) q[7];
cx q[6],q[7];
ry(1.2839668858783053) q[6];
ry(2.4084623849584177) q[7];
cx q[6],q[7];
ry(-2.5014592002488905) q[0];
ry(1.6827895292813437) q[2];
cx q[0],q[2];
ry(0.46455472312101515) q[0];
ry(-2.2738975683330303) q[2];
cx q[0],q[2];
ry(0.16504289920093257) q[2];
ry(-1.6706523561802316) q[4];
cx q[2],q[4];
ry(2.16078412988828) q[2];
ry(-2.9932766170513623) q[4];
cx q[2],q[4];
ry(-1.972512725146613) q[4];
ry(-1.428822331940924) q[6];
cx q[4],q[6];
ry(-3.0120012991694867) q[4];
ry(-2.5938379320151594) q[6];
cx q[4],q[6];
ry(2.4556071529554155) q[1];
ry(-1.0358180430023698) q[3];
cx q[1],q[3];
ry(2.624068636030371) q[1];
ry(-1.2105518447011536) q[3];
cx q[1],q[3];
ry(-1.397571685158438) q[3];
ry(-0.6595527668899709) q[5];
cx q[3],q[5];
ry(-1.3501511224485754) q[3];
ry(0.3890116866321677) q[5];
cx q[3],q[5];
ry(3.023959909258982) q[5];
ry(-1.972121135572137) q[7];
cx q[5],q[7];
ry(-0.2906155107807493) q[5];
ry(-1.2460245240908023) q[7];
cx q[5],q[7];
ry(0.4796757112722103) q[0];
ry(3.0789210072654396) q[3];
cx q[0],q[3];
ry(1.7552369094577625) q[0];
ry(-0.33851012021061816) q[3];
cx q[0],q[3];
ry(1.1599128089771797) q[1];
ry(-1.0306799927259522) q[2];
cx q[1],q[2];
ry(-2.8660483937838355) q[1];
ry(-0.9787268896404742) q[2];
cx q[1],q[2];
ry(1.3014866523361095) q[2];
ry(-3.0389581249223783) q[5];
cx q[2],q[5];
ry(3.012213297777119) q[2];
ry(-0.4864867864573559) q[5];
cx q[2],q[5];
ry(3.0585725057688204) q[3];
ry(-1.667218744255757) q[4];
cx q[3],q[4];
ry(-2.904807854284669) q[3];
ry(-0.13843519080283606) q[4];
cx q[3],q[4];
ry(-1.2945688683494518) q[4];
ry(-2.5487925227186343) q[7];
cx q[4],q[7];
ry(-3.109640924468597) q[4];
ry(-0.4381034434190365) q[7];
cx q[4],q[7];
ry(0.14140101843901473) q[5];
ry(3.0857958486212214) q[6];
cx q[5],q[6];
ry(1.2699983090801483) q[5];
ry(0.3278530510612373) q[6];
cx q[5],q[6];
ry(-2.7009725910243967) q[0];
ry(-2.186842591805881) q[1];
cx q[0],q[1];
ry(2.7828827858083796) q[0];
ry(2.3854394446715665) q[1];
cx q[0],q[1];
ry(0.4717891828145056) q[2];
ry(0.7094742998854215) q[3];
cx q[2],q[3];
ry(-1.2736825615858889) q[2];
ry(1.03731892045374) q[3];
cx q[2],q[3];
ry(-2.1141979508501922) q[4];
ry(0.45225053331290077) q[5];
cx q[4],q[5];
ry(1.3241935702014977) q[4];
ry(2.6684450446324552) q[5];
cx q[4],q[5];
ry(2.343406070374093) q[6];
ry(2.1562553670086397) q[7];
cx q[6],q[7];
ry(1.0291525696820578) q[6];
ry(0.9420257711497747) q[7];
cx q[6],q[7];
ry(0.39006816398216904) q[0];
ry(2.010167612031915) q[2];
cx q[0],q[2];
ry(2.2197223714150045) q[0];
ry(1.434061828513259) q[2];
cx q[0],q[2];
ry(-2.871658934583204) q[2];
ry(2.463931257946544) q[4];
cx q[2],q[4];
ry(-1.9958964106671848) q[2];
ry(0.5445798647742208) q[4];
cx q[2],q[4];
ry(-3.09423578338668) q[4];
ry(1.3809450114799997) q[6];
cx q[4],q[6];
ry(2.3457055785499796) q[4];
ry(1.7699237898514928) q[6];
cx q[4],q[6];
ry(0.194433006280744) q[1];
ry(1.88122265334469) q[3];
cx q[1],q[3];
ry(-2.145544418516077) q[1];
ry(1.264394529955647) q[3];
cx q[1],q[3];
ry(-2.424839648819146) q[3];
ry(2.616031835943867) q[5];
cx q[3],q[5];
ry(-2.1458620268153994) q[3];
ry(1.1945894513954574) q[5];
cx q[3],q[5];
ry(-1.5224167428147588) q[5];
ry(-1.3330392585102973) q[7];
cx q[5],q[7];
ry(2.5794174959530327) q[5];
ry(-2.238635968728813) q[7];
cx q[5],q[7];
ry(1.6523412084402354) q[0];
ry(-0.9998751231129726) q[3];
cx q[0],q[3];
ry(-1.1305238116991596) q[0];
ry(-0.836190109190091) q[3];
cx q[0],q[3];
ry(-2.986057708179621) q[1];
ry(0.4003393517200324) q[2];
cx q[1],q[2];
ry(-0.521275876402762) q[1];
ry(1.7659922397082874) q[2];
cx q[1],q[2];
ry(-1.6523021001009273) q[2];
ry(2.425555761438004) q[5];
cx q[2],q[5];
ry(-1.2424334900818839) q[2];
ry(-0.5187616103798846) q[5];
cx q[2],q[5];
ry(-2.5278781028731467) q[3];
ry(1.943063114823178) q[4];
cx q[3],q[4];
ry(1.4157052099743357) q[3];
ry(0.0978995911066828) q[4];
cx q[3],q[4];
ry(0.31935650373405355) q[4];
ry(0.3705637790855182) q[7];
cx q[4],q[7];
ry(-0.28383495626403477) q[4];
ry(-3.0285965458072153) q[7];
cx q[4],q[7];
ry(2.4550934409531107) q[5];
ry(0.10406346725021898) q[6];
cx q[5],q[6];
ry(2.5641371762284497) q[5];
ry(0.26887232897011604) q[6];
cx q[5],q[6];
ry(1.6025726805172338) q[0];
ry(-1.2841786572778053) q[1];
cx q[0],q[1];
ry(0.6607614317552846) q[0];
ry(1.9100147290018417) q[1];
cx q[0],q[1];
ry(-0.6522866816824316) q[2];
ry(3.130360066338434) q[3];
cx q[2],q[3];
ry(-2.8502845466381728) q[2];
ry(-0.6767053607086206) q[3];
cx q[2],q[3];
ry(0.546049471763741) q[4];
ry(-2.2939839309162617) q[5];
cx q[4],q[5];
ry(-1.7866203843954558) q[4];
ry(-0.6813004123409624) q[5];
cx q[4],q[5];
ry(-0.7849411347404739) q[6];
ry(3.0856473398033173) q[7];
cx q[6],q[7];
ry(-2.5024366303356884) q[6];
ry(-1.288188501624628) q[7];
cx q[6],q[7];
ry(-0.2022128712977196) q[0];
ry(-0.3260474091681608) q[2];
cx q[0],q[2];
ry(-2.6448871465815627) q[0];
ry(-0.4934483512166495) q[2];
cx q[0],q[2];
ry(-1.3686184589025974) q[2];
ry(-2.084155483674362) q[4];
cx q[2],q[4];
ry(-1.3281611260737538) q[2];
ry(-2.344279343731096) q[4];
cx q[2],q[4];
ry(-1.1858856106806888) q[4];
ry(-2.0849062074718843) q[6];
cx q[4],q[6];
ry(3.045878742955393) q[4];
ry(-3.126354714854252) q[6];
cx q[4],q[6];
ry(1.2300156808580145) q[1];
ry(0.11348077221949654) q[3];
cx q[1],q[3];
ry(-0.2586811197061612) q[1];
ry(-0.46116462110007467) q[3];
cx q[1],q[3];
ry(-2.664409149984976) q[3];
ry(1.2556317347963573) q[5];
cx q[3],q[5];
ry(2.1294161576060073) q[3];
ry(0.45886689353275967) q[5];
cx q[3],q[5];
ry(3.001706497351106) q[5];
ry(0.8282863291863878) q[7];
cx q[5],q[7];
ry(2.453247344357249) q[5];
ry(-2.7209997079685584) q[7];
cx q[5],q[7];
ry(0.3990435766489114) q[0];
ry(-3.096478459072698) q[3];
cx q[0],q[3];
ry(-2.4912898791127627) q[0];
ry(-2.0366554996209647) q[3];
cx q[0],q[3];
ry(0.8047498666588226) q[1];
ry(-2.5010528289539296) q[2];
cx q[1],q[2];
ry(-0.6957216003143386) q[1];
ry(0.7378238113670247) q[2];
cx q[1],q[2];
ry(-1.98000758453175) q[2];
ry(-0.11766768337553835) q[5];
cx q[2],q[5];
ry(1.0234224310126279) q[2];
ry(-0.018040947995021202) q[5];
cx q[2],q[5];
ry(2.590248649423473) q[3];
ry(-2.059268990207552) q[4];
cx q[3],q[4];
ry(-0.9884395582587915) q[3];
ry(1.0843816947828178) q[4];
cx q[3],q[4];
ry(3.0016160958085782) q[4];
ry(1.428850302395745) q[7];
cx q[4],q[7];
ry(2.0801640054528376) q[4];
ry(-1.633237014849839) q[7];
cx q[4],q[7];
ry(-1.4829384912384764) q[5];
ry(2.114903240551369) q[6];
cx q[5],q[6];
ry(2.430375000306476) q[5];
ry(-1.7604085246799333) q[6];
cx q[5],q[6];
ry(0.6282891984507677) q[0];
ry(2.4718341095983223) q[1];
cx q[0],q[1];
ry(2.1920305393458364) q[0];
ry(-2.246498881808409) q[1];
cx q[0],q[1];
ry(0.18296148697260772) q[2];
ry(-0.7966542929077027) q[3];
cx q[2],q[3];
ry(0.804192240071214) q[2];
ry(-1.958173235989821) q[3];
cx q[2],q[3];
ry(0.361856555049089) q[4];
ry(-0.700261447566568) q[5];
cx q[4],q[5];
ry(1.0942776495962832) q[4];
ry(-2.301416888100237) q[5];
cx q[4],q[5];
ry(0.10633877604692453) q[6];
ry(1.4085484012253784) q[7];
cx q[6],q[7];
ry(0.4440680164475852) q[6];
ry(-0.26586761259474156) q[7];
cx q[6],q[7];
ry(2.485890322206879) q[0];
ry(2.928905339912232) q[2];
cx q[0],q[2];
ry(-2.94229243459249) q[0];
ry(-0.10909563302101598) q[2];
cx q[0],q[2];
ry(-1.3854493910857217) q[2];
ry(-2.099864257138247) q[4];
cx q[2],q[4];
ry(-0.6687306477657735) q[2];
ry(0.4767525446760067) q[4];
cx q[2],q[4];
ry(-0.668323093724383) q[4];
ry(2.1557428684995514) q[6];
cx q[4],q[6];
ry(-2.820853406891111) q[4];
ry(-0.15757943088729567) q[6];
cx q[4],q[6];
ry(-2.9156767493370115) q[1];
ry(2.2328025605798922) q[3];
cx q[1],q[3];
ry(-0.7846054668905672) q[1];
ry(1.0294512925961046) q[3];
cx q[1],q[3];
ry(-0.39966334861337904) q[3];
ry(-1.3011612416703597) q[5];
cx q[3],q[5];
ry(-1.68225140966892) q[3];
ry(2.5166673583345154) q[5];
cx q[3],q[5];
ry(-2.7102369791886836) q[5];
ry(-0.8510585089928835) q[7];
cx q[5],q[7];
ry(0.7213766631688623) q[5];
ry(1.0066692305258522) q[7];
cx q[5],q[7];
ry(2.481951009307288) q[0];
ry(-1.162704407331577) q[3];
cx q[0],q[3];
ry(-2.2463602160246987) q[0];
ry(-0.3109741462062761) q[3];
cx q[0],q[3];
ry(2.524465114615317) q[1];
ry(1.4340275024028957) q[2];
cx q[1],q[2];
ry(-0.4401881030126587) q[1];
ry(1.3550859744881851) q[2];
cx q[1],q[2];
ry(-2.3625495812259514) q[2];
ry(-2.3703752480681235) q[5];
cx q[2],q[5];
ry(-2.6125755298169735) q[2];
ry(2.662474802748339) q[5];
cx q[2],q[5];
ry(-2.185965200271408) q[3];
ry(-1.0672020911357674) q[4];
cx q[3],q[4];
ry(1.7818497126260384) q[3];
ry(1.0495043359242118) q[4];
cx q[3],q[4];
ry(-0.22897316484299424) q[4];
ry(-2.3463692711671937) q[7];
cx q[4],q[7];
ry(-2.9329986190304247) q[4];
ry(1.9212514590717504) q[7];
cx q[4],q[7];
ry(2.3440494465614417) q[5];
ry(2.2623739327374617) q[6];
cx q[5],q[6];
ry(1.4317520773546868) q[5];
ry(-0.0478180044233199) q[6];
cx q[5],q[6];
ry(-0.5497281139298974) q[0];
ry(2.1100932887756727) q[1];
cx q[0],q[1];
ry(-0.39121565027669036) q[0];
ry(-1.5388579609390685) q[1];
cx q[0],q[1];
ry(2.377954443614985) q[2];
ry(-1.9598755052860248) q[3];
cx q[2],q[3];
ry(-2.2859200977511183) q[2];
ry(-0.03843440506185036) q[3];
cx q[2],q[3];
ry(-1.7048649923373436) q[4];
ry(-0.3835384926476565) q[5];
cx q[4],q[5];
ry(2.915012043534412) q[4];
ry(3.029371126644626) q[5];
cx q[4],q[5];
ry(2.968658814125218) q[6];
ry(-0.6254122378459772) q[7];
cx q[6],q[7];
ry(-2.7033974119790756) q[6];
ry(2.4469509028670995) q[7];
cx q[6],q[7];
ry(-3.1290173796432597) q[0];
ry(1.3442953994879066) q[2];
cx q[0],q[2];
ry(0.8280033909186537) q[0];
ry(2.9632264873323444) q[2];
cx q[0],q[2];
ry(-1.458782292069282) q[2];
ry(1.1268623940722722) q[4];
cx q[2],q[4];
ry(3.136544967900799) q[2];
ry(2.140416101911484) q[4];
cx q[2],q[4];
ry(-0.6249804768826043) q[4];
ry(-1.5499798371678708) q[6];
cx q[4],q[6];
ry(1.3395339441671863) q[4];
ry(0.08269990400147502) q[6];
cx q[4],q[6];
ry(2.5246065631333283) q[1];
ry(0.8608657402040817) q[3];
cx q[1],q[3];
ry(2.925138816368843) q[1];
ry(0.8350254724095049) q[3];
cx q[1],q[3];
ry(0.9609360684239729) q[3];
ry(-1.2556693204927376) q[5];
cx q[3],q[5];
ry(-0.5393603928990045) q[3];
ry(-2.7866206718906583) q[5];
cx q[3],q[5];
ry(2.330523775146128) q[5];
ry(2.434622002375677) q[7];
cx q[5],q[7];
ry(2.2754425771157747) q[5];
ry(-2.4878846281517246) q[7];
cx q[5],q[7];
ry(2.900280661346594) q[0];
ry(2.7676138583379157) q[3];
cx q[0],q[3];
ry(2.9687854917262455) q[0];
ry(0.6718263619352101) q[3];
cx q[0],q[3];
ry(-2.3051563150030354) q[1];
ry(-0.9924495297265691) q[2];
cx q[1],q[2];
ry(-2.8177079440483985) q[1];
ry(0.3816415399160764) q[2];
cx q[1],q[2];
ry(-0.8488737512756567) q[2];
ry(-1.7961237163792179) q[5];
cx q[2],q[5];
ry(0.5496981718427821) q[2];
ry(0.04618429151741754) q[5];
cx q[2],q[5];
ry(-1.8122256599131377) q[3];
ry(0.014101891051911508) q[4];
cx q[3],q[4];
ry(1.7267814916233615) q[3];
ry(-1.768972382657981) q[4];
cx q[3],q[4];
ry(-0.763477773535588) q[4];
ry(0.2308029622295533) q[7];
cx q[4],q[7];
ry(-2.9694090007103813) q[4];
ry(0.8612424773544147) q[7];
cx q[4],q[7];
ry(1.184488167700029) q[5];
ry(2.2368949657729953) q[6];
cx q[5],q[6];
ry(0.7381845185503122) q[5];
ry(1.642094245669445) q[6];
cx q[5],q[6];
ry(2.9940832862701807) q[0];
ry(0.052906552614456544) q[1];
cx q[0],q[1];
ry(0.8462718409050235) q[0];
ry(0.9591238756908373) q[1];
cx q[0],q[1];
ry(-1.228202916584629) q[2];
ry(0.6623340941939206) q[3];
cx q[2],q[3];
ry(-1.2243740999688526) q[2];
ry(-1.3502791061932218) q[3];
cx q[2],q[3];
ry(2.1940761301443605) q[4];
ry(0.490307922430035) q[5];
cx q[4],q[5];
ry(1.1010912834450215) q[4];
ry(-2.4754043305058415) q[5];
cx q[4],q[5];
ry(-2.658317480378269) q[6];
ry(-1.3260134872711573) q[7];
cx q[6],q[7];
ry(2.3687360901608736) q[6];
ry(2.6286116810594087) q[7];
cx q[6],q[7];
ry(-0.9167925303044893) q[0];
ry(-2.871448397397532) q[2];
cx q[0],q[2];
ry(3.0209515983865396) q[0];
ry(2.3813137328991) q[2];
cx q[0],q[2];
ry(-1.7961353136697369) q[2];
ry(2.642119452438478) q[4];
cx q[2],q[4];
ry(2.623575180821872) q[2];
ry(2.661668845386973) q[4];
cx q[2],q[4];
ry(0.7418028460994863) q[4];
ry(1.2522316897581256) q[6];
cx q[4],q[6];
ry(0.39256251750996896) q[4];
ry(-3.0030070770980646) q[6];
cx q[4],q[6];
ry(-2.41543897986019) q[1];
ry(-0.16952080205196263) q[3];
cx q[1],q[3];
ry(-2.1830385971588386) q[1];
ry(-2.0309768723116965) q[3];
cx q[1],q[3];
ry(-1.9965119311126065) q[3];
ry(1.1620304483976387) q[5];
cx q[3],q[5];
ry(1.5349657872670872) q[3];
ry(-0.741472345345775) q[5];
cx q[3],q[5];
ry(-1.9034081877340858) q[5];
ry(-0.9743154015477591) q[7];
cx q[5],q[7];
ry(-1.2938762052867414) q[5];
ry(-1.244968764580906) q[7];
cx q[5],q[7];
ry(-1.660187631859512) q[0];
ry(1.4348307572884496) q[3];
cx q[0],q[3];
ry(-2.939435063870767) q[0];
ry(-0.7976383671281537) q[3];
cx q[0],q[3];
ry(2.902144011462398) q[1];
ry(-0.12608090321458154) q[2];
cx q[1],q[2];
ry(1.6504911836920098) q[1];
ry(-1.9680039713030708) q[2];
cx q[1],q[2];
ry(0.9511610565499904) q[2];
ry(2.39206529252629) q[5];
cx q[2],q[5];
ry(3.092271625078546) q[2];
ry(2.5655281347500063) q[5];
cx q[2],q[5];
ry(0.7516476978564963) q[3];
ry(-1.6658277846792355) q[4];
cx q[3],q[4];
ry(-2.696570681111188) q[3];
ry(-2.199572933397527) q[4];
cx q[3],q[4];
ry(-2.9528734897917452) q[4];
ry(0.9031957510445793) q[7];
cx q[4],q[7];
ry(1.6971664652144132) q[4];
ry(1.0517206497862093) q[7];
cx q[4],q[7];
ry(2.2603448858542663) q[5];
ry(-2.8766684148980897) q[6];
cx q[5],q[6];
ry(-0.5371556465676331) q[5];
ry(2.1895667374235233) q[6];
cx q[5],q[6];
ry(-1.5262763744195258) q[0];
ry(-1.1558663494780141) q[1];
cx q[0],q[1];
ry(-1.629354415119293) q[0];
ry(-1.5374532799110505) q[1];
cx q[0],q[1];
ry(2.85123618099884) q[2];
ry(0.5674830678038381) q[3];
cx q[2],q[3];
ry(0.6931746358963551) q[2];
ry(-1.6920417304591164) q[3];
cx q[2],q[3];
ry(0.5166167165646502) q[4];
ry(1.8698947463980204) q[5];
cx q[4],q[5];
ry(-1.7938334086093441) q[4];
ry(2.3366570204238744) q[5];
cx q[4],q[5];
ry(-3.1262969810795225) q[6];
ry(-1.7084439391986073) q[7];
cx q[6],q[7];
ry(-2.10447747512616) q[6];
ry(-0.010978836031530827) q[7];
cx q[6],q[7];
ry(3.0004956659618043) q[0];
ry(-2.4670627832961967) q[2];
cx q[0],q[2];
ry(2.428318990034619) q[0];
ry(-1.0824694560476373) q[2];
cx q[0],q[2];
ry(0.07450243841777356) q[2];
ry(-3.07154904315632) q[4];
cx q[2],q[4];
ry(-1.9197045116430647) q[2];
ry(0.8743399707756607) q[4];
cx q[2],q[4];
ry(-2.4184920747042398) q[4];
ry(2.4107260597139475) q[6];
cx q[4],q[6];
ry(-1.4282892924069701) q[4];
ry(-2.8491505331684364) q[6];
cx q[4],q[6];
ry(1.4601760774757144) q[1];
ry(1.4820905689939412) q[3];
cx q[1],q[3];
ry(-1.0444728336378977) q[1];
ry(-2.6667699709362083) q[3];
cx q[1],q[3];
ry(-2.0929953044101604) q[3];
ry(-2.3778061535646295) q[5];
cx q[3],q[5];
ry(-1.705463085770613) q[3];
ry(-2.267615029697867) q[5];
cx q[3],q[5];
ry(-0.1827129407509842) q[5];
ry(-1.8568836334611962) q[7];
cx q[5],q[7];
ry(-0.7136444985170654) q[5];
ry(1.4362339238141197) q[7];
cx q[5],q[7];
ry(0.6571613250176975) q[0];
ry(0.07154604992079872) q[3];
cx q[0],q[3];
ry(-1.2174409340637589) q[0];
ry(0.5833841172241695) q[3];
cx q[0],q[3];
ry(-0.20937732176336257) q[1];
ry(-2.574565977305215) q[2];
cx q[1],q[2];
ry(0.986223381016516) q[1];
ry(0.6772067287088399) q[2];
cx q[1],q[2];
ry(-0.7380621904326033) q[2];
ry(2.3942982019431236) q[5];
cx q[2],q[5];
ry(-1.0184921894796477) q[2];
ry(3.0209328099839663) q[5];
cx q[2],q[5];
ry(1.769007250148321) q[3];
ry(1.31841688207929) q[4];
cx q[3],q[4];
ry(-2.7005597361372606) q[3];
ry(-0.07551228974713807) q[4];
cx q[3],q[4];
ry(3.05098857692864) q[4];
ry(-0.22740975893902524) q[7];
cx q[4],q[7];
ry(-2.500693300825152) q[4];
ry(2.6344482407549883) q[7];
cx q[4],q[7];
ry(0.0073423761926632025) q[5];
ry(0.5517644831640949) q[6];
cx q[5],q[6];
ry(0.11521370774918549) q[5];
ry(-1.880278281855582) q[6];
cx q[5],q[6];
ry(-1.2290818139265727) q[0];
ry(2.542332354985377) q[1];
cx q[0],q[1];
ry(1.7238675015006395) q[0];
ry(2.1117744735713933) q[1];
cx q[0],q[1];
ry(-2.834534742687653) q[2];
ry(-2.9290068468995965) q[3];
cx q[2],q[3];
ry(2.7858759423133352) q[2];
ry(-2.736610458170574) q[3];
cx q[2],q[3];
ry(-0.5518345189346332) q[4];
ry(-1.357851311719321) q[5];
cx q[4],q[5];
ry(2.5128857468272034) q[4];
ry(1.330349847279586) q[5];
cx q[4],q[5];
ry(1.8555086937303684) q[6];
ry(-1.6780659278500414) q[7];
cx q[6],q[7];
ry(0.5938349023289097) q[6];
ry(0.7935719852218865) q[7];
cx q[6],q[7];
ry(0.827515714093777) q[0];
ry(-1.8657288031826686) q[2];
cx q[0],q[2];
ry(-0.5110802672074808) q[0];
ry(2.953158769645464) q[2];
cx q[0],q[2];
ry(2.3591249419791622) q[2];
ry(2.6138962914527135) q[4];
cx q[2],q[4];
ry(0.3322241378724812) q[2];
ry(0.6468514439037726) q[4];
cx q[2],q[4];
ry(0.3497314242038837) q[4];
ry(0.4466474944963911) q[6];
cx q[4],q[6];
ry(1.7602157943711099) q[4];
ry(-3.072395808530993) q[6];
cx q[4],q[6];
ry(-2.3673652066393767) q[1];
ry(-0.5368947749509179) q[3];
cx q[1],q[3];
ry(0.1044009446893357) q[1];
ry(0.28473081472735373) q[3];
cx q[1],q[3];
ry(2.476972829087276) q[3];
ry(1.7264959704262992) q[5];
cx q[3],q[5];
ry(0.8098775543446344) q[3];
ry(-1.8237409691800104) q[5];
cx q[3],q[5];
ry(-1.860874272353042) q[5];
ry(2.6162094756040783) q[7];
cx q[5],q[7];
ry(2.6232604686821155) q[5];
ry(-1.9140335752223732) q[7];
cx q[5],q[7];
ry(1.8476193269992494) q[0];
ry(2.903585641254789) q[3];
cx q[0],q[3];
ry(1.509390288423785) q[0];
ry(-2.115696390428998) q[3];
cx q[0],q[3];
ry(-1.5503814713730186) q[1];
ry(-2.9886728354239325) q[2];
cx q[1],q[2];
ry(0.5202627283164905) q[1];
ry(-1.9106486502164646) q[2];
cx q[1],q[2];
ry(-1.1654082275035826) q[2];
ry(0.1471585874648751) q[5];
cx q[2],q[5];
ry(1.0626146514308876) q[2];
ry(1.7739759381410758) q[5];
cx q[2],q[5];
ry(2.6088740698685187) q[3];
ry(2.321896422339237) q[4];
cx q[3],q[4];
ry(-0.7797287024146017) q[3];
ry(-0.7074059580381755) q[4];
cx q[3],q[4];
ry(0.7544740679440722) q[4];
ry(1.3188073808565717) q[7];
cx q[4],q[7];
ry(-2.8665396153013796) q[4];
ry(2.6053916975042997) q[7];
cx q[4],q[7];
ry(-1.2636550098804715) q[5];
ry(-0.8465043777833979) q[6];
cx q[5],q[6];
ry(-2.3210573651312223) q[5];
ry(-0.19349515136166584) q[6];
cx q[5],q[6];
ry(-0.3864558480239273) q[0];
ry(0.5104955252622367) q[1];
cx q[0],q[1];
ry(2.9560322863104895) q[0];
ry(-2.070437720682081) q[1];
cx q[0],q[1];
ry(-0.1568140061300496) q[2];
ry(-0.7360649144597392) q[3];
cx q[2],q[3];
ry(0.5973472009799033) q[2];
ry(2.348393829574272) q[3];
cx q[2],q[3];
ry(-1.4702607404790742) q[4];
ry(1.2073464448002933) q[5];
cx q[4],q[5];
ry(-0.5723876314015053) q[4];
ry(0.08289920484488977) q[5];
cx q[4],q[5];
ry(-0.6760965071581557) q[6];
ry(2.28362928649591) q[7];
cx q[6],q[7];
ry(0.41725555214701426) q[6];
ry(2.992896155847383) q[7];
cx q[6],q[7];
ry(-1.2651992279261508) q[0];
ry(2.1511348595383395) q[2];
cx q[0],q[2];
ry(-1.1249960064902276) q[0];
ry(3.1098045758614994) q[2];
cx q[0],q[2];
ry(-2.7288445191945137) q[2];
ry(-2.0923253419534866) q[4];
cx q[2],q[4];
ry(0.36128045259697084) q[2];
ry(1.199962136647118) q[4];
cx q[2],q[4];
ry(-2.264858176646941) q[4];
ry(-0.8834130127450379) q[6];
cx q[4],q[6];
ry(2.552052080541125) q[4];
ry(-0.14140902701216085) q[6];
cx q[4],q[6];
ry(-2.40968485600198) q[1];
ry(1.0245721784309703) q[3];
cx q[1],q[3];
ry(2.7136906337122646) q[1];
ry(2.249206272391648) q[3];
cx q[1],q[3];
ry(1.8857743012383024) q[3];
ry(2.244048782067269) q[5];
cx q[3],q[5];
ry(-0.9428429683125197) q[3];
ry(2.7557051607459084) q[5];
cx q[3],q[5];
ry(2.115517223203819) q[5];
ry(-1.5265705735979243) q[7];
cx q[5],q[7];
ry(-2.438595883743114) q[5];
ry(-1.1407344416713237) q[7];
cx q[5],q[7];
ry(-1.69093314205327) q[0];
ry(0.7869471958955502) q[3];
cx q[0],q[3];
ry(-1.8730421324424944) q[0];
ry(2.321302145547503) q[3];
cx q[0],q[3];
ry(-1.1492941839163233) q[1];
ry(-1.9084686395231347) q[2];
cx q[1],q[2];
ry(-0.9234644261720427) q[1];
ry(0.19263719709582272) q[2];
cx q[1],q[2];
ry(2.0488371738258735) q[2];
ry(1.3875180043766497) q[5];
cx q[2],q[5];
ry(-0.7308148105256901) q[2];
ry(2.7187903463655303) q[5];
cx q[2],q[5];
ry(-0.09233438369158263) q[3];
ry(0.9319772153828848) q[4];
cx q[3],q[4];
ry(2.7941575758934376) q[3];
ry(-2.7688121907895256) q[4];
cx q[3],q[4];
ry(0.0973793309670723) q[4];
ry(2.909914742243096) q[7];
cx q[4],q[7];
ry(-0.44627885972176223) q[4];
ry(2.2615259774942245) q[7];
cx q[4],q[7];
ry(1.2027299039982957) q[5];
ry(-0.31965160114574154) q[6];
cx q[5],q[6];
ry(2.225495981630177) q[5];
ry(-2.9453722947667873) q[6];
cx q[5],q[6];
ry(3.1137519171800205) q[0];
ry(0.5115679495829795) q[1];
cx q[0],q[1];
ry(-0.9163913036885134) q[0];
ry(-0.49046969057547685) q[1];
cx q[0],q[1];
ry(1.0883597078585723) q[2];
ry(2.20679986509072) q[3];
cx q[2],q[3];
ry(-2.8629107582575615) q[2];
ry(1.6270369797507591) q[3];
cx q[2],q[3];
ry(0.7864521608422556) q[4];
ry(-1.982901204163995) q[5];
cx q[4],q[5];
ry(2.0134307848875284) q[4];
ry(1.1691836833957598) q[5];
cx q[4],q[5];
ry(2.570819216278175) q[6];
ry(2.8901372665103584) q[7];
cx q[6],q[7];
ry(-0.6765451047197218) q[6];
ry(-2.5644266572572754) q[7];
cx q[6],q[7];
ry(-0.010700674664437138) q[0];
ry(2.874385425521833) q[2];
cx q[0],q[2];
ry(0.3830046182372558) q[0];
ry(0.21297025225368138) q[2];
cx q[0],q[2];
ry(0.6190272943683028) q[2];
ry(1.1834596153612191) q[4];
cx q[2],q[4];
ry(2.889255459101483) q[2];
ry(2.7257765467868085) q[4];
cx q[2],q[4];
ry(-2.7394780561345065) q[4];
ry(2.3163727904644342) q[6];
cx q[4],q[6];
ry(2.6503797255338832) q[4];
ry(-1.0479941402985058) q[6];
cx q[4],q[6];
ry(0.9952885710804423) q[1];
ry(0.24090820730621296) q[3];
cx q[1],q[3];
ry(-0.12481566245829523) q[1];
ry(2.329447759795788) q[3];
cx q[1],q[3];
ry(-0.7592779777351693) q[3];
ry(-2.5010798893317907) q[5];
cx q[3],q[5];
ry(1.1975841012630681) q[3];
ry(0.8209483526113929) q[5];
cx q[3],q[5];
ry(2.81271015343432) q[5];
ry(-1.7404182346767305) q[7];
cx q[5],q[7];
ry(0.2409911459877554) q[5];
ry(0.42004346526189185) q[7];
cx q[5],q[7];
ry(0.7028720485182689) q[0];
ry(0.659205993830075) q[3];
cx q[0],q[3];
ry(0.899323630770287) q[0];
ry(1.966618630054108) q[3];
cx q[0],q[3];
ry(1.5779324635166123) q[1];
ry(2.427854529625856) q[2];
cx q[1],q[2];
ry(-1.2300336892746255) q[1];
ry(2.5543068497200623) q[2];
cx q[1],q[2];
ry(2.0883570591218703) q[2];
ry(0.5155430887112851) q[5];
cx q[2],q[5];
ry(-0.09234543929951755) q[2];
ry(1.9136836915509292) q[5];
cx q[2],q[5];
ry(2.1907894082559594) q[3];
ry(-1.912197178853174) q[4];
cx q[3],q[4];
ry(0.8939404091320995) q[3];
ry(-1.608819516171292) q[4];
cx q[3],q[4];
ry(1.1333871305854017) q[4];
ry(-0.8967373339277493) q[7];
cx q[4],q[7];
ry(1.3985334621809358) q[4];
ry(1.3599696803913908) q[7];
cx q[4],q[7];
ry(1.463749375833709) q[5];
ry(1.9598000486311709) q[6];
cx q[5],q[6];
ry(2.157869745234273) q[5];
ry(0.10046805837833349) q[6];
cx q[5],q[6];
ry(2.401299780581154) q[0];
ry(0.7304010578724077) q[1];
ry(-2.972689690743958) q[2];
ry(0.00518884219256055) q[3];
ry(1.6347320855794898) q[4];
ry(2.203835599903142) q[5];
ry(-0.5217566324287245) q[6];
ry(2.726145909708854) q[7];