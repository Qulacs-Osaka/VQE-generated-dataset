OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.1399260915280172) q[0];
ry(-2.4812511645471576) q[1];
cx q[0],q[1];
ry(-0.33046618506035963) q[0];
ry(-2.4474366830656833) q[1];
cx q[0],q[1];
ry(1.9551693661196972) q[0];
ry(0.9635189389795888) q[2];
cx q[0],q[2];
ry(0.21042723119177387) q[0];
ry(2.0790590116685865) q[2];
cx q[0],q[2];
ry(2.2087983058042315) q[0];
ry(2.44141187577407) q[3];
cx q[0],q[3];
ry(-3.129423798553324) q[0];
ry(-0.8117405922363021) q[3];
cx q[0],q[3];
ry(-2.761312526402087) q[1];
ry(1.5029581508730618) q[2];
cx q[1],q[2];
ry(2.642833621076713) q[1];
ry(1.6709914718189491) q[2];
cx q[1],q[2];
ry(1.187215566741357) q[1];
ry(1.5516826598556626) q[3];
cx q[1],q[3];
ry(2.5155738194696418) q[1];
ry(0.19633872469823818) q[3];
cx q[1],q[3];
ry(-1.725996197468096) q[2];
ry(-1.7185291734925192) q[3];
cx q[2],q[3];
ry(2.5641346338535618) q[2];
ry(1.489765519892092) q[3];
cx q[2],q[3];
ry(3.1350088812760144) q[0];
ry(-2.39772664265476) q[1];
cx q[0],q[1];
ry(2.6805700419163228) q[0];
ry(2.359997226957569) q[1];
cx q[0],q[1];
ry(0.8288316508722628) q[0];
ry(1.940991723414876) q[2];
cx q[0],q[2];
ry(0.2024302063851602) q[0];
ry(-0.9776478591458218) q[2];
cx q[0],q[2];
ry(-0.7114797428907131) q[0];
ry(-0.11451590273610704) q[3];
cx q[0],q[3];
ry(1.2629300342951966) q[0];
ry(-2.2845611953979117) q[3];
cx q[0],q[3];
ry(0.7676371363712278) q[1];
ry(0.8172352793641985) q[2];
cx q[1],q[2];
ry(-1.8127782343452783) q[1];
ry(-1.8421615552847044) q[2];
cx q[1],q[2];
ry(-1.9681640485565701) q[1];
ry(-3.007957007153465) q[3];
cx q[1],q[3];
ry(-1.1981925570058376) q[1];
ry(1.2370710730194734) q[3];
cx q[1],q[3];
ry(-2.113087437090864) q[2];
ry(-2.862964240732912) q[3];
cx q[2],q[3];
ry(0.25760898368665996) q[2];
ry(-1.7005754500811576) q[3];
cx q[2],q[3];
ry(0.9997902001194799) q[0];
ry(-3.0665784717630955) q[1];
cx q[0],q[1];
ry(-2.7867680925257403) q[0];
ry(2.9828328992975477) q[1];
cx q[0],q[1];
ry(-0.5068799853183998) q[0];
ry(-1.9911206727202062) q[2];
cx q[0],q[2];
ry(2.3317196366098947) q[0];
ry(-2.417654622634206) q[2];
cx q[0],q[2];
ry(-0.7558287555817141) q[0];
ry(-1.5911675094864997) q[3];
cx q[0],q[3];
ry(-2.479899756708134) q[0];
ry(1.7645904570550508) q[3];
cx q[0],q[3];
ry(-0.6270667536830326) q[1];
ry(-2.573685842106048) q[2];
cx q[1],q[2];
ry(0.9260352745627235) q[1];
ry(2.45362549048878) q[2];
cx q[1],q[2];
ry(-1.9599798007657183) q[1];
ry(2.5493376085786634) q[3];
cx q[1],q[3];
ry(1.1543025085615142) q[1];
ry(-0.9594716436561663) q[3];
cx q[1],q[3];
ry(-1.8062596726469708) q[2];
ry(-0.38692203451477464) q[3];
cx q[2],q[3];
ry(-0.45309186209043073) q[2];
ry(-1.2778808166954168) q[3];
cx q[2],q[3];
ry(2.4117476638261888) q[0];
ry(-2.9316152547045284) q[1];
cx q[0],q[1];
ry(2.4043113520014616) q[0];
ry(2.1943472852570225) q[1];
cx q[0],q[1];
ry(-2.4941005294722336) q[0];
ry(2.488871388313137) q[2];
cx q[0],q[2];
ry(-2.8833901831781525) q[0];
ry(-0.7321172295695799) q[2];
cx q[0],q[2];
ry(1.5727741956963428) q[0];
ry(2.7314326739995978) q[3];
cx q[0],q[3];
ry(-2.2531522111895192) q[0];
ry(2.3916714838153785) q[3];
cx q[0],q[3];
ry(3.057845597431582) q[1];
ry(-0.16604900150423596) q[2];
cx q[1],q[2];
ry(2.2840553185517916) q[1];
ry(-1.7025293566638666) q[2];
cx q[1],q[2];
ry(2.2386221152138344) q[1];
ry(-0.672157827585564) q[3];
cx q[1],q[3];
ry(1.198506692899012) q[1];
ry(-2.0000673052126077) q[3];
cx q[1],q[3];
ry(-2.527088195112197) q[2];
ry(1.7996302211062485) q[3];
cx q[2],q[3];
ry(1.6543871141451012) q[2];
ry(2.3217128038354855) q[3];
cx q[2],q[3];
ry(-0.7014708890043053) q[0];
ry(2.1367133179876827) q[1];
cx q[0],q[1];
ry(-1.8048120804251422) q[0];
ry(0.6281811252056722) q[1];
cx q[0],q[1];
ry(-2.621677250072371) q[0];
ry(-0.8078767147812052) q[2];
cx q[0],q[2];
ry(-2.7151275301701405) q[0];
ry(1.093670979016955) q[2];
cx q[0],q[2];
ry(-1.3369831054684074) q[0];
ry(0.009596866110820555) q[3];
cx q[0],q[3];
ry(-2.147437093936058) q[0];
ry(0.46808941126323855) q[3];
cx q[0],q[3];
ry(1.0764584583578831) q[1];
ry(1.216670790769765) q[2];
cx q[1],q[2];
ry(2.7213322893733607) q[1];
ry(2.8037226135351427) q[2];
cx q[1],q[2];
ry(-0.8746835164000012) q[1];
ry(1.2474907796482917) q[3];
cx q[1],q[3];
ry(0.8696443903516831) q[1];
ry(-0.6098206031809311) q[3];
cx q[1],q[3];
ry(2.5422386726445256) q[2];
ry(-0.62342458164082) q[3];
cx q[2],q[3];
ry(1.191250855892226) q[2];
ry(-1.774310099572899) q[3];
cx q[2],q[3];
ry(-2.898524121767304) q[0];
ry(2.5620119031539943) q[1];
cx q[0],q[1];
ry(0.837220397158942) q[0];
ry(-1.8281481346614088) q[1];
cx q[0],q[1];
ry(0.17349421584949187) q[0];
ry(-2.6710896711559373) q[2];
cx q[0],q[2];
ry(2.029825556945374) q[0];
ry(-2.782534719059785) q[2];
cx q[0],q[2];
ry(2.116784515503517) q[0];
ry(2.506806999461311) q[3];
cx q[0],q[3];
ry(1.601420404882586) q[0];
ry(-0.9011913201566004) q[3];
cx q[0],q[3];
ry(-1.9565190880835543) q[1];
ry(3.1373403590623536) q[2];
cx q[1],q[2];
ry(-0.4230728098893577) q[1];
ry(-1.7497959771489293) q[2];
cx q[1],q[2];
ry(2.756354566675711) q[1];
ry(2.8452284539822053) q[3];
cx q[1],q[3];
ry(-2.0472594531921233) q[1];
ry(1.4001234848469764) q[3];
cx q[1],q[3];
ry(2.992484527899724) q[2];
ry(-2.549252234401543) q[3];
cx q[2],q[3];
ry(-2.458747641399385) q[2];
ry(-0.5834176934697022) q[3];
cx q[2],q[3];
ry(-2.5080356691752073) q[0];
ry(-0.9288805462626231) q[1];
cx q[0],q[1];
ry(0.7748397617823886) q[0];
ry(2.2756594779845103) q[1];
cx q[0],q[1];
ry(-2.5998577544838053) q[0];
ry(1.351543530803954) q[2];
cx q[0],q[2];
ry(-1.8751986626010764) q[0];
ry(-1.0350989070657297) q[2];
cx q[0],q[2];
ry(-0.3988161824938457) q[0];
ry(-3.072561267830786) q[3];
cx q[0],q[3];
ry(-2.944402298202657) q[0];
ry(1.1225924698096734) q[3];
cx q[0],q[3];
ry(3.138722673164227) q[1];
ry(-1.12664926883693) q[2];
cx q[1],q[2];
ry(0.14174767789803422) q[1];
ry(-0.7053096618455825) q[2];
cx q[1],q[2];
ry(0.44731890285055265) q[1];
ry(0.6527725617238325) q[3];
cx q[1],q[3];
ry(-0.8078479186154556) q[1];
ry(2.4049667691553918) q[3];
cx q[1],q[3];
ry(2.1016138317197397) q[2];
ry(-2.793823093673816) q[3];
cx q[2],q[3];
ry(-3.0905702812334366) q[2];
ry(-0.5529483287901173) q[3];
cx q[2],q[3];
ry(-0.9353914886761157) q[0];
ry(-3.1389449338697326) q[1];
cx q[0],q[1];
ry(1.7636907317934938) q[0];
ry(-1.9731123603427863) q[1];
cx q[0],q[1];
ry(-1.8509373389373551) q[0];
ry(-1.3171640744519992) q[2];
cx q[0],q[2];
ry(-1.4947944080303) q[0];
ry(1.080036580984645) q[2];
cx q[0],q[2];
ry(-2.3850426883649876) q[0];
ry(2.5784623439522405) q[3];
cx q[0],q[3];
ry(-1.9739677967946074) q[0];
ry(1.5289858071984648) q[3];
cx q[0],q[3];
ry(0.5537494984747102) q[1];
ry(0.5301498548650594) q[2];
cx q[1],q[2];
ry(-2.449349736908071) q[1];
ry(-0.07534090054577725) q[2];
cx q[1],q[2];
ry(1.0751888320609035) q[1];
ry(-0.540347038168763) q[3];
cx q[1],q[3];
ry(1.0921481313809682) q[1];
ry(-2.377596712643743) q[3];
cx q[1],q[3];
ry(-0.15671133778614324) q[2];
ry(1.9662638131316132) q[3];
cx q[2],q[3];
ry(1.6857753926006076) q[2];
ry(2.693245992596247) q[3];
cx q[2],q[3];
ry(-0.8325110399705307) q[0];
ry(-1.50763447354933) q[1];
cx q[0],q[1];
ry(1.1311702344395709) q[0];
ry(-0.05418472868351553) q[1];
cx q[0],q[1];
ry(0.05642054890733039) q[0];
ry(0.2442259050433213) q[2];
cx q[0],q[2];
ry(-1.8680427770261723) q[0];
ry(2.619450476042325) q[2];
cx q[0],q[2];
ry(2.1751360672121995) q[0];
ry(1.1538422923321436) q[3];
cx q[0],q[3];
ry(-1.8828749546029613) q[0];
ry(1.3123989055918108) q[3];
cx q[0],q[3];
ry(-0.3861039280486549) q[1];
ry(-1.7316395244254466) q[2];
cx q[1],q[2];
ry(-0.592923641935279) q[1];
ry(-2.864809865022469) q[2];
cx q[1],q[2];
ry(-1.1283886739191127) q[1];
ry(-2.6309634325526856) q[3];
cx q[1],q[3];
ry(0.7708157662226718) q[1];
ry(-0.7324948824745715) q[3];
cx q[1],q[3];
ry(-2.974878290032755) q[2];
ry(-2.189224692783209) q[3];
cx q[2],q[3];
ry(-2.802784823818246) q[2];
ry(0.2577570361884325) q[3];
cx q[2],q[3];
ry(2.1279877257251805) q[0];
ry(0.03504132970729934) q[1];
cx q[0],q[1];
ry(-0.31314304694989215) q[0];
ry(2.41342793401441) q[1];
cx q[0],q[1];
ry(-2.262494026472329) q[0];
ry(2.1234383509455195) q[2];
cx q[0],q[2];
ry(2.0405225025158638) q[0];
ry(-0.5858869322889719) q[2];
cx q[0],q[2];
ry(0.1712523522403776) q[0];
ry(3.040743848720281) q[3];
cx q[0],q[3];
ry(2.3786575450571084) q[0];
ry(-2.673041523257779) q[3];
cx q[0],q[3];
ry(-0.4118961056595021) q[1];
ry(-2.457499908853994) q[2];
cx q[1],q[2];
ry(-0.34272819961018275) q[1];
ry(-0.20617343047664694) q[2];
cx q[1],q[2];
ry(2.9023028254288934) q[1];
ry(1.8869772332417547) q[3];
cx q[1],q[3];
ry(1.7736629759881777) q[1];
ry(2.8559554304947623) q[3];
cx q[1],q[3];
ry(0.13085436816187815) q[2];
ry(0.3504689797755083) q[3];
cx q[2],q[3];
ry(2.474454900724663) q[2];
ry(1.0154598849782488) q[3];
cx q[2],q[3];
ry(2.608749014382813) q[0];
ry(0.8672511659369784) q[1];
cx q[0],q[1];
ry(1.4809824938787832) q[0];
ry(1.2163794301856923) q[1];
cx q[0],q[1];
ry(0.8233006596765886) q[0];
ry(-1.7458321168734214) q[2];
cx q[0],q[2];
ry(-0.8903234555193935) q[0];
ry(-2.7629142618918983) q[2];
cx q[0],q[2];
ry(0.2800047517538227) q[0];
ry(1.5891098257280594) q[3];
cx q[0],q[3];
ry(2.975323401028934) q[0];
ry(-2.796628772243471) q[3];
cx q[0],q[3];
ry(-0.12921273950818257) q[1];
ry(-2.2155181406590563) q[2];
cx q[1],q[2];
ry(-2.6754836641296844) q[1];
ry(1.5721203730723294) q[2];
cx q[1],q[2];
ry(0.9765481307374921) q[1];
ry(-1.8030928186007564) q[3];
cx q[1],q[3];
ry(-1.3863433536494556) q[1];
ry(0.8637744298885685) q[3];
cx q[1],q[3];
ry(-1.3212011076324328) q[2];
ry(-2.999197923614058) q[3];
cx q[2],q[3];
ry(0.4910653905889858) q[2];
ry(1.689379818699153) q[3];
cx q[2],q[3];
ry(-2.1438390725223866) q[0];
ry(-1.69640957956259) q[1];
cx q[0],q[1];
ry(-2.389997476108077) q[0];
ry(0.02741693686033564) q[1];
cx q[0],q[1];
ry(-0.13940880680530174) q[0];
ry(-1.4946658165969922) q[2];
cx q[0],q[2];
ry(-0.2095088066976502) q[0];
ry(-2.3147401259274387) q[2];
cx q[0],q[2];
ry(-1.9241324591109121) q[0];
ry(-1.249279782394027) q[3];
cx q[0],q[3];
ry(-0.10731354036624907) q[0];
ry(-0.29253805761307716) q[3];
cx q[0],q[3];
ry(0.00146865375985153) q[1];
ry(-1.530113370667637) q[2];
cx q[1],q[2];
ry(-2.6049818545565033) q[1];
ry(2.702141910693759) q[2];
cx q[1],q[2];
ry(2.9767499346064596) q[1];
ry(-0.7062438393893511) q[3];
cx q[1],q[3];
ry(0.8377645322650433) q[1];
ry(2.1587451553599353) q[3];
cx q[1],q[3];
ry(0.9281174584279835) q[2];
ry(0.9821902367186466) q[3];
cx q[2],q[3];
ry(2.150125688199825) q[2];
ry(1.1012151725177857) q[3];
cx q[2],q[3];
ry(0.09557351548912417) q[0];
ry(-2.1799916535152386) q[1];
cx q[0],q[1];
ry(-2.3545490086525622) q[0];
ry(1.273255208914118) q[1];
cx q[0],q[1];
ry(1.8658753782858029) q[0];
ry(-0.46028382450718824) q[2];
cx q[0],q[2];
ry(0.44456131134717336) q[0];
ry(3.0659151021777653) q[2];
cx q[0],q[2];
ry(-2.431616658123531) q[0];
ry(-2.610616578324842) q[3];
cx q[0],q[3];
ry(2.7141563957798955) q[0];
ry(-2.0517277463246355) q[3];
cx q[0],q[3];
ry(2.2502713245097805) q[1];
ry(-1.4609988785529051) q[2];
cx q[1],q[2];
ry(-1.4484632000895068) q[1];
ry(-0.41776925944821025) q[2];
cx q[1],q[2];
ry(-0.015268436983643587) q[1];
ry(-1.7978683274080671) q[3];
cx q[1],q[3];
ry(-1.0608125879966768) q[1];
ry(-1.8570533928366268) q[3];
cx q[1],q[3];
ry(1.9619942319360115) q[2];
ry(1.3635497810170927) q[3];
cx q[2],q[3];
ry(-0.9366826073090646) q[2];
ry(1.4966431053753189) q[3];
cx q[2],q[3];
ry(-2.6335460065349645) q[0];
ry(-2.266217042663733) q[1];
cx q[0],q[1];
ry(2.2411565440583736) q[0];
ry(1.9390230738843028) q[1];
cx q[0],q[1];
ry(2.4588901108219288) q[0];
ry(1.0253510525372027) q[2];
cx q[0],q[2];
ry(0.07389872658760943) q[0];
ry(-1.4957773148549143) q[2];
cx q[0],q[2];
ry(0.5232801742543289) q[0];
ry(1.5284816888981716) q[3];
cx q[0],q[3];
ry(-2.814310831019093) q[0];
ry(1.7149261470662367) q[3];
cx q[0],q[3];
ry(0.10493752932216213) q[1];
ry(-1.853939804822724) q[2];
cx q[1],q[2];
ry(-2.8122667120118368) q[1];
ry(2.571539906464084) q[2];
cx q[1],q[2];
ry(-0.7183101503590201) q[1];
ry(-0.8432353745151945) q[3];
cx q[1],q[3];
ry(1.884522423840223) q[1];
ry(0.8225235728351166) q[3];
cx q[1],q[3];
ry(1.9107803193764807) q[2];
ry(-2.5235834453361328) q[3];
cx q[2],q[3];
ry(2.1509806335527237) q[2];
ry(0.5684034029933729) q[3];
cx q[2],q[3];
ry(-1.1574045027777355) q[0];
ry(-1.629150285520521) q[1];
cx q[0],q[1];
ry(0.3629093439905793) q[0];
ry(-2.3198437232247393) q[1];
cx q[0],q[1];
ry(1.5422351441044786) q[0];
ry(2.9187627363898367) q[2];
cx q[0],q[2];
ry(-2.512710396438071) q[0];
ry(-2.957472198151642) q[2];
cx q[0],q[2];
ry(2.9168054207679277) q[0];
ry(-0.6589728636359516) q[3];
cx q[0],q[3];
ry(-1.7662237726562724) q[0];
ry(2.038785888987655) q[3];
cx q[0],q[3];
ry(2.893290892470461) q[1];
ry(0.6499103494058316) q[2];
cx q[1],q[2];
ry(-1.675199158076554) q[1];
ry(2.913629547212102) q[2];
cx q[1],q[2];
ry(2.9202205871746942) q[1];
ry(2.256285670916607) q[3];
cx q[1],q[3];
ry(-2.5491467628510245) q[1];
ry(0.40711310691080893) q[3];
cx q[1],q[3];
ry(-0.4519229046889911) q[2];
ry(-1.692232189561351) q[3];
cx q[2],q[3];
ry(-0.690423141252965) q[2];
ry(-0.8650502782448637) q[3];
cx q[2],q[3];
ry(2.847433058233096) q[0];
ry(-3.086391345692808) q[1];
cx q[0],q[1];
ry(-1.5654167115460402) q[0];
ry(-0.9469192425862141) q[1];
cx q[0],q[1];
ry(0.6178017640454659) q[0];
ry(3.0216089038337) q[2];
cx q[0],q[2];
ry(2.633271338927408) q[0];
ry(-0.8755438003358433) q[2];
cx q[0],q[2];
ry(-2.4843641745066933) q[0];
ry(2.2232840830900527) q[3];
cx q[0],q[3];
ry(-1.0569085468704549) q[0];
ry(-1.8469716096944568) q[3];
cx q[0],q[3];
ry(-1.985694466784068) q[1];
ry(-1.6338224050465904) q[2];
cx q[1],q[2];
ry(1.310147281479633) q[1];
ry(-0.9420718340162973) q[2];
cx q[1],q[2];
ry(1.9182408989566513) q[1];
ry(-0.1769629546683399) q[3];
cx q[1],q[3];
ry(2.609448602083474) q[1];
ry(-1.1387332536956742) q[3];
cx q[1],q[3];
ry(-0.12628222069431216) q[2];
ry(2.777845985848751) q[3];
cx q[2],q[3];
ry(-1.1885500567584775) q[2];
ry(-2.7493302630543845) q[3];
cx q[2],q[3];
ry(2.9720015607344354) q[0];
ry(-0.11165135890830148) q[1];
cx q[0],q[1];
ry(1.007177431763651) q[0];
ry(-1.175735656483329) q[1];
cx q[0],q[1];
ry(-1.8288567984522468) q[0];
ry(-3.048297571571533) q[2];
cx q[0],q[2];
ry(-1.3211698024085203) q[0];
ry(2.758076852530631) q[2];
cx q[0],q[2];
ry(-2.4778817422476687) q[0];
ry(1.0080467304885314) q[3];
cx q[0],q[3];
ry(1.7634212891412242) q[0];
ry(1.6951938374844167) q[3];
cx q[0],q[3];
ry(2.1263680433634176) q[1];
ry(-2.6508053706950165) q[2];
cx q[1],q[2];
ry(1.4309677729659764) q[1];
ry(-2.707308204577342) q[2];
cx q[1],q[2];
ry(2.232565489627728) q[1];
ry(-0.22074455878651822) q[3];
cx q[1],q[3];
ry(-1.0900215316595954) q[1];
ry(1.280969767234005) q[3];
cx q[1],q[3];
ry(0.633187341672588) q[2];
ry(-2.8899020916790055) q[3];
cx q[2],q[3];
ry(-1.6496452687576184) q[2];
ry(-2.7302157509599323) q[3];
cx q[2],q[3];
ry(3.127741567763898) q[0];
ry(2.9750362259310457) q[1];
cx q[0],q[1];
ry(2.337767071200984) q[0];
ry(-0.8681401322119306) q[1];
cx q[0],q[1];
ry(-1.546125553049145) q[0];
ry(-2.963053683489844) q[2];
cx q[0],q[2];
ry(0.45134210202778274) q[0];
ry(1.7251692882852154) q[2];
cx q[0],q[2];
ry(-0.728191966484408) q[0];
ry(-1.6606891352412925) q[3];
cx q[0],q[3];
ry(1.6473707961390525) q[0];
ry(0.3893507638884497) q[3];
cx q[0],q[3];
ry(2.7389760510128087) q[1];
ry(1.9007551632717667) q[2];
cx q[1],q[2];
ry(2.176935909638896) q[1];
ry(1.8921765884451773) q[2];
cx q[1],q[2];
ry(-1.275719616905982) q[1];
ry(-2.5154681743354335) q[3];
cx q[1],q[3];
ry(-1.589254956893285) q[1];
ry(0.7539819717386319) q[3];
cx q[1],q[3];
ry(-1.0760377685137623) q[2];
ry(-1.2929969498712157) q[3];
cx q[2],q[3];
ry(-0.7226029599830233) q[2];
ry(-2.483287069078275) q[3];
cx q[2],q[3];
ry(2.4155654940582165) q[0];
ry(-0.4094161605825725) q[1];
cx q[0],q[1];
ry(-2.3396950251530626) q[0];
ry(-2.969466928449906) q[1];
cx q[0],q[1];
ry(-1.9699204779406543) q[0];
ry(-1.948173406001378) q[2];
cx q[0],q[2];
ry(2.2489959728262443) q[0];
ry(1.001459792168337) q[2];
cx q[0],q[2];
ry(-1.9910460142239712) q[0];
ry(0.8246174244782019) q[3];
cx q[0],q[3];
ry(-0.930274298196303) q[0];
ry(0.8169042383833518) q[3];
cx q[0],q[3];
ry(2.0768090635953103) q[1];
ry(1.8946156953757427) q[2];
cx q[1],q[2];
ry(3.120589523069411) q[1];
ry(-2.383075653678164) q[2];
cx q[1],q[2];
ry(-0.5745923365306478) q[1];
ry(2.2574229433173736) q[3];
cx q[1],q[3];
ry(-0.9428590884671184) q[1];
ry(1.1734191278807744) q[3];
cx q[1],q[3];
ry(1.3671337782699666) q[2];
ry(-2.395316784272826) q[3];
cx q[2],q[3];
ry(-0.569545195824606) q[2];
ry(0.1739832356634361) q[3];
cx q[2],q[3];
ry(3.114717951732306) q[0];
ry(2.9592669228229087) q[1];
cx q[0],q[1];
ry(0.5449764455436699) q[0];
ry(-0.9219188008844245) q[1];
cx q[0],q[1];
ry(1.2340258109514264) q[0];
ry(-2.323817698401084) q[2];
cx q[0],q[2];
ry(-1.4999142197645732) q[0];
ry(-1.3266677882899813) q[2];
cx q[0],q[2];
ry(-2.9985040012678623) q[0];
ry(-1.9657237372959582) q[3];
cx q[0],q[3];
ry(-0.31025069124954546) q[0];
ry(-0.1928106943326992) q[3];
cx q[0],q[3];
ry(-0.40833624236054167) q[1];
ry(-0.06271979141275885) q[2];
cx q[1],q[2];
ry(2.442239738079545) q[1];
ry(2.3066803431022667) q[2];
cx q[1],q[2];
ry(-0.2927741825213417) q[1];
ry(-0.39568453966858286) q[3];
cx q[1],q[3];
ry(1.419280741473042) q[1];
ry(-0.5868563156351874) q[3];
cx q[1],q[3];
ry(-2.899072506614101) q[2];
ry(1.5821903389201182) q[3];
cx q[2],q[3];
ry(2.79691141844941) q[2];
ry(1.3550247188282523) q[3];
cx q[2],q[3];
ry(-1.705431367332819) q[0];
ry(2.920211038729403) q[1];
ry(0.35850209885279594) q[2];
ry(0.7833903284490669) q[3];