OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.777244076464318) q[0];
ry(-2.483093207882393) q[1];
cx q[0],q[1];
ry(-2.4662574088412064) q[0];
ry(2.831486048138) q[1];
cx q[0],q[1];
ry(-1.4430453931789138) q[1];
ry(-0.8338088960495263) q[2];
cx q[1],q[2];
ry(-0.4326504712088841) q[1];
ry(-0.022765492651975805) q[2];
cx q[1],q[2];
ry(-1.1346494802387976) q[2];
ry(0.24517690115700885) q[3];
cx q[2],q[3];
ry(0.5948172563447409) q[2];
ry(-2.019397529360667) q[3];
cx q[2],q[3];
ry(-3.0262479674719494) q[3];
ry(-1.6611970881801108) q[4];
cx q[3],q[4];
ry(-2.419461667830701) q[3];
ry(2.1359653327659265) q[4];
cx q[3],q[4];
ry(-0.26154123611742724) q[4];
ry(0.6278580110159098) q[5];
cx q[4],q[5];
ry(-1.6498342902369942) q[4];
ry(0.7992397651151615) q[5];
cx q[4],q[5];
ry(-1.3673970571673095) q[5];
ry(0.7784804243919101) q[6];
cx q[5],q[6];
ry(2.4352990856730394) q[5];
ry(0.019525624152840315) q[6];
cx q[5],q[6];
ry(-1.8042154980718599) q[6];
ry(-2.518232693623102) q[7];
cx q[6],q[7];
ry(0.4012419701644733) q[6];
ry(1.5404783719075015) q[7];
cx q[6],q[7];
ry(-1.7902319040551664) q[0];
ry(0.07952148094045253) q[1];
cx q[0],q[1];
ry(-2.668224213666517) q[0];
ry(-1.2012278376306753) q[1];
cx q[0],q[1];
ry(1.4029427165988382) q[1];
ry(-1.6479441462848872) q[2];
cx q[1],q[2];
ry(-0.3470334624054576) q[1];
ry(1.728576461370502) q[2];
cx q[1],q[2];
ry(0.4616454767125389) q[2];
ry(0.6552406773782335) q[3];
cx q[2],q[3];
ry(-2.692903623461888) q[2];
ry(-2.8733383014695435) q[3];
cx q[2],q[3];
ry(-0.7962849756307389) q[3];
ry(-2.4487379369183886) q[4];
cx q[3],q[4];
ry(-1.0506571022412068) q[3];
ry(0.40152306514653235) q[4];
cx q[3],q[4];
ry(-0.25338299841760925) q[4];
ry(-0.1353576108525072) q[5];
cx q[4],q[5];
ry(-0.9884982251056975) q[4];
ry(0.9872320892444657) q[5];
cx q[4],q[5];
ry(-0.4347137049364454) q[5];
ry(-0.34660858015058066) q[6];
cx q[5],q[6];
ry(1.4728727312187424) q[5];
ry(-2.8470787764599255) q[6];
cx q[5],q[6];
ry(-0.5735278137231941) q[6];
ry(2.025827540660827) q[7];
cx q[6],q[7];
ry(2.205138013103005) q[6];
ry(2.023809554848863) q[7];
cx q[6],q[7];
ry(-2.685323749732308) q[0];
ry(1.3273748060921158) q[1];
cx q[0],q[1];
ry(-1.8327915039378602) q[0];
ry(-2.982804593225985) q[1];
cx q[0],q[1];
ry(-0.6331322649282204) q[1];
ry(0.050036453351176666) q[2];
cx q[1],q[2];
ry(-0.7539214076861223) q[1];
ry(0.09259231005532342) q[2];
cx q[1],q[2];
ry(-0.9601300557018781) q[2];
ry(2.6447338683903103) q[3];
cx q[2],q[3];
ry(-1.6024299436871356) q[2];
ry(0.6508559124594804) q[3];
cx q[2],q[3];
ry(-0.9904353869719347) q[3];
ry(-2.6172992263468267) q[4];
cx q[3],q[4];
ry(-1.2824113537716597) q[3];
ry(-1.5821438619362445) q[4];
cx q[3],q[4];
ry(2.7497033947267484) q[4];
ry(1.4060305627998828) q[5];
cx q[4],q[5];
ry(2.7442691204219156) q[4];
ry(3.0146041115056157) q[5];
cx q[4],q[5];
ry(-1.3131436618915977) q[5];
ry(-1.9475069275943326) q[6];
cx q[5],q[6];
ry(-2.66175376949221) q[5];
ry(-2.9314334930491444) q[6];
cx q[5],q[6];
ry(-0.055371462909432484) q[6];
ry(1.5713952867734005) q[7];
cx q[6],q[7];
ry(1.019889963698482) q[6];
ry(-0.06831280495907262) q[7];
cx q[6],q[7];
ry(-2.9174584288737266) q[0];
ry(2.2914477572185645) q[1];
cx q[0],q[1];
ry(-0.3275978320180137) q[0];
ry(-0.056855313079899085) q[1];
cx q[0],q[1];
ry(-0.07987919887361716) q[1];
ry(0.20305237121831343) q[2];
cx q[1],q[2];
ry(-0.3322628344769499) q[1];
ry(1.675813104378604) q[2];
cx q[1],q[2];
ry(-2.0656241430921334) q[2];
ry(2.8566267733518935) q[3];
cx q[2],q[3];
ry(-0.007274920612477429) q[2];
ry(-0.6109089646068124) q[3];
cx q[2],q[3];
ry(2.3334532539945196) q[3];
ry(-0.9738148811286909) q[4];
cx q[3],q[4];
ry(1.9741336488551469) q[3];
ry(0.28083107798921975) q[4];
cx q[3],q[4];
ry(-1.590026989475091) q[4];
ry(2.130358040391343) q[5];
cx q[4],q[5];
ry(-1.9309370737654943) q[4];
ry(1.393551761915449) q[5];
cx q[4],q[5];
ry(-2.323954432227236) q[5];
ry(2.3909883018265896) q[6];
cx q[5],q[6];
ry(-0.4193721876563186) q[5];
ry(3.012260952011025) q[6];
cx q[5],q[6];
ry(1.9145852262205258) q[6];
ry(2.7367646183044223) q[7];
cx q[6],q[7];
ry(0.3639613745562338) q[6];
ry(2.7980457853621363) q[7];
cx q[6],q[7];
ry(-2.634005298625795) q[0];
ry(-1.5708132755980826) q[1];
cx q[0],q[1];
ry(-2.4011506318387505) q[0];
ry(1.2317694202006386) q[1];
cx q[0],q[1];
ry(2.2352384098489457) q[1];
ry(1.7943506286477433) q[2];
cx q[1],q[2];
ry(-2.6652667756844126) q[1];
ry(2.775238889065788) q[2];
cx q[1],q[2];
ry(-1.1675041488666) q[2];
ry(1.6945724731362342) q[3];
cx q[2],q[3];
ry(0.445780905549222) q[2];
ry(2.1819719913574396) q[3];
cx q[2],q[3];
ry(-1.1800149050811717) q[3];
ry(-0.8367264632868894) q[4];
cx q[3],q[4];
ry(2.270014968695519) q[3];
ry(-1.0386842851492564) q[4];
cx q[3],q[4];
ry(2.0021637760406144) q[4];
ry(-3.099309024461706) q[5];
cx q[4],q[5];
ry(-0.509644044986815) q[4];
ry(-1.2808331937705608) q[5];
cx q[4],q[5];
ry(-2.1283909376571746) q[5];
ry(2.638865352367418) q[6];
cx q[5],q[6];
ry(-1.1794192089303337) q[5];
ry(2.2292353582332867) q[6];
cx q[5],q[6];
ry(0.0028850144340159816) q[6];
ry(2.9213793603648246) q[7];
cx q[6],q[7];
ry(-1.3530106055442621) q[6];
ry(2.627686191896378) q[7];
cx q[6],q[7];
ry(2.7470242259720394) q[0];
ry(0.08663055571558687) q[1];
cx q[0],q[1];
ry(2.1415284602172493) q[0];
ry(1.4260986288680337) q[1];
cx q[0],q[1];
ry(-2.5718457044126977) q[1];
ry(2.041284153897822) q[2];
cx q[1],q[2];
ry(0.8950790246836062) q[1];
ry(1.6484022709754529) q[2];
cx q[1],q[2];
ry(-0.7572823192349709) q[2];
ry(0.6318257347590561) q[3];
cx q[2],q[3];
ry(2.4546473965667186) q[2];
ry(-2.7216038681585624) q[3];
cx q[2],q[3];
ry(0.7610576943992002) q[3];
ry(3.0913544113920044) q[4];
cx q[3],q[4];
ry(-1.1596289290734452) q[3];
ry(-2.463457783242743) q[4];
cx q[3],q[4];
ry(-0.2471396634221934) q[4];
ry(-3.0090658430310504) q[5];
cx q[4],q[5];
ry(-1.1565046759787254) q[4];
ry(2.537316519048807) q[5];
cx q[4],q[5];
ry(2.11642800531315) q[5];
ry(-2.4463556549648553) q[6];
cx q[5],q[6];
ry(-0.5376412492100329) q[5];
ry(2.097984190529081) q[6];
cx q[5],q[6];
ry(0.14653073484311907) q[6];
ry(3.015123027586659) q[7];
cx q[6],q[7];
ry(3.113339292957408) q[6];
ry(-2.8105288050492923) q[7];
cx q[6],q[7];
ry(1.0623175109033094) q[0];
ry(0.007749519661034654) q[1];
cx q[0],q[1];
ry(1.2837326477036215) q[0];
ry(-1.8326792407332366) q[1];
cx q[0],q[1];
ry(-0.8842319552620301) q[1];
ry(-1.1518253404624372) q[2];
cx q[1],q[2];
ry(-2.9933540226518214) q[1];
ry(0.9904223667045056) q[2];
cx q[1],q[2];
ry(0.4209075597359444) q[2];
ry(-2.0645551618856013) q[3];
cx q[2],q[3];
ry(2.1723178295318926) q[2];
ry(-1.4357546955610445) q[3];
cx q[2],q[3];
ry(-2.281669488237964) q[3];
ry(-1.9951401858956355) q[4];
cx q[3],q[4];
ry(0.07995653075386411) q[3];
ry(-1.9091321753815658) q[4];
cx q[3],q[4];
ry(-2.0808385719905065) q[4];
ry(-0.04486801364566251) q[5];
cx q[4],q[5];
ry(1.5099675622033046) q[4];
ry(-0.48286881724180397) q[5];
cx q[4],q[5];
ry(1.1011662793108572) q[5];
ry(-0.012719206137886485) q[6];
cx q[5],q[6];
ry(-2.513472559332415) q[5];
ry(-0.0880482816253275) q[6];
cx q[5],q[6];
ry(-0.8731045038109925) q[6];
ry(0.7494073633697509) q[7];
cx q[6],q[7];
ry(-2.415929917957558) q[6];
ry(1.4344926434875296) q[7];
cx q[6],q[7];
ry(1.0611906103495905) q[0];
ry(-1.9972368383852794) q[1];
cx q[0],q[1];
ry(-2.7413322954068784) q[0];
ry(-2.0089789731916) q[1];
cx q[0],q[1];
ry(0.7180455245592441) q[1];
ry(1.1395862535570886) q[2];
cx q[1],q[2];
ry(1.5189954387554592) q[1];
ry(-1.322505518811668) q[2];
cx q[1],q[2];
ry(1.9923395927387928) q[2];
ry(0.4126027940084045) q[3];
cx q[2],q[3];
ry(-2.4179184030929974) q[2];
ry(-2.4847192627609567) q[3];
cx q[2],q[3];
ry(2.5846810150589343) q[3];
ry(0.982303790737084) q[4];
cx q[3],q[4];
ry(2.261579069944661) q[3];
ry(-1.7029462581418375) q[4];
cx q[3],q[4];
ry(-1.1582137484844557) q[4];
ry(-2.453318164617551) q[5];
cx q[4],q[5];
ry(0.8060215844502734) q[4];
ry(1.5106501198278162) q[5];
cx q[4],q[5];
ry(-2.909042113172639) q[5];
ry(0.725673468711558) q[6];
cx q[5],q[6];
ry(0.3698149975527373) q[5];
ry(-1.601912113259143) q[6];
cx q[5],q[6];
ry(2.8108453633422874) q[6];
ry(1.9546516868488055) q[7];
cx q[6],q[7];
ry(0.5375765157916765) q[6];
ry(0.5320399123504388) q[7];
cx q[6],q[7];
ry(-0.12939923659386032) q[0];
ry(2.5474032109554936) q[1];
cx q[0],q[1];
ry(-1.3572348955244165) q[0];
ry(-2.05726906205647) q[1];
cx q[0],q[1];
ry(-1.8336309376762614) q[1];
ry(1.615913833499859) q[2];
cx q[1],q[2];
ry(2.071411585205641) q[1];
ry(1.9663237502636015) q[2];
cx q[1],q[2];
ry(-1.0547041610533536) q[2];
ry(1.3338827748607907) q[3];
cx q[2],q[3];
ry(2.231901738532866) q[2];
ry(-2.5570871052292765) q[3];
cx q[2],q[3];
ry(-0.5285421174656414) q[3];
ry(-0.3585900870113701) q[4];
cx q[3],q[4];
ry(0.5724670455861123) q[3];
ry(-2.27408798010043) q[4];
cx q[3],q[4];
ry(-1.7547466279757142) q[4];
ry(0.9027992157337001) q[5];
cx q[4],q[5];
ry(0.04269249962310795) q[4];
ry(-2.5863818313342906) q[5];
cx q[4],q[5];
ry(-1.8639631019672507) q[5];
ry(-1.1652018802902182) q[6];
cx q[5],q[6];
ry(-1.4931208321683656) q[5];
ry(2.3299302799501223) q[6];
cx q[5],q[6];
ry(-0.6883108822381416) q[6];
ry(-1.7021543052350048) q[7];
cx q[6],q[7];
ry(-0.1228422666942741) q[6];
ry(0.3479865405246718) q[7];
cx q[6],q[7];
ry(1.2528544343601848) q[0];
ry(0.48998014623736985) q[1];
cx q[0],q[1];
ry(0.06265721474599152) q[0];
ry(2.946851829700104) q[1];
cx q[0],q[1];
ry(1.6177203418421904) q[1];
ry(-0.8726021474772203) q[2];
cx q[1],q[2];
ry(-2.2528202975443063) q[1];
ry(1.9146334418896451) q[2];
cx q[1],q[2];
ry(1.8918629850349173) q[2];
ry(-0.9777120465507795) q[3];
cx q[2],q[3];
ry(2.6268033227245557) q[2];
ry(0.5763807589974399) q[3];
cx q[2],q[3];
ry(2.271442348214645) q[3];
ry(-2.831389721478467) q[4];
cx q[3],q[4];
ry(-1.5763222583873358) q[3];
ry(2.2017302666961447) q[4];
cx q[3],q[4];
ry(-2.5243890098034734) q[4];
ry(-2.3293896991591905) q[5];
cx q[4],q[5];
ry(1.186886376092808) q[4];
ry(-2.8962004027306207) q[5];
cx q[4],q[5];
ry(-1.4725001053033917) q[5];
ry(1.3205073413040944) q[6];
cx q[5],q[6];
ry(-0.7734180070402292) q[5];
ry(-0.6966879630818594) q[6];
cx q[5],q[6];
ry(2.6947597986428544) q[6];
ry(1.0667144894708178) q[7];
cx q[6],q[7];
ry(1.4817982026498093) q[6];
ry(-0.5705465186851775) q[7];
cx q[6],q[7];
ry(-2.2661998815345785) q[0];
ry(1.0957594876704484) q[1];
cx q[0],q[1];
ry(-2.089668729808726) q[0];
ry(-2.173668326176915) q[1];
cx q[0],q[1];
ry(2.872447208442509) q[1];
ry(-3.0694677589213084) q[2];
cx q[1],q[2];
ry(2.383138590852416) q[1];
ry(3.0216985477904457) q[2];
cx q[1],q[2];
ry(1.6847121514352983) q[2];
ry(0.4398024414255969) q[3];
cx q[2],q[3];
ry(2.903022299563073) q[2];
ry(0.7051115079509856) q[3];
cx q[2],q[3];
ry(1.5901685202634148) q[3];
ry(2.8466739431557393) q[4];
cx q[3],q[4];
ry(-0.1755147491315187) q[3];
ry(-2.94132108596691) q[4];
cx q[3],q[4];
ry(0.10360666723616523) q[4];
ry(2.038065609612163) q[5];
cx q[4],q[5];
ry(-0.18128501539658792) q[4];
ry(0.819293887032471) q[5];
cx q[4],q[5];
ry(2.2155081647291297) q[5];
ry(1.500302531456438) q[6];
cx q[5],q[6];
ry(0.7763299999442063) q[5];
ry(-3.08505024867545) q[6];
cx q[5],q[6];
ry(0.16991635935479896) q[6];
ry(-2.880386946100468) q[7];
cx q[6],q[7];
ry(0.6959554088062966) q[6];
ry(-0.08848852717062528) q[7];
cx q[6],q[7];
ry(2.549760200493444) q[0];
ry(-2.0979642847248776) q[1];
cx q[0],q[1];
ry(0.5601089033161752) q[0];
ry(3.110790695378394) q[1];
cx q[0],q[1];
ry(-1.0290782950315591) q[1];
ry(2.2236221587486362) q[2];
cx q[1],q[2];
ry(-2.6803130412218032) q[1];
ry(-1.0330519617807683) q[2];
cx q[1],q[2];
ry(0.5294446434240365) q[2];
ry(-1.0095249406662186) q[3];
cx q[2],q[3];
ry(-1.1299529572144147) q[2];
ry(0.4384038443325279) q[3];
cx q[2],q[3];
ry(0.2873784971675071) q[3];
ry(0.4946199353997312) q[4];
cx q[3],q[4];
ry(1.3991594057083463) q[3];
ry(-1.9718637435931923) q[4];
cx q[3],q[4];
ry(-0.35748920948594154) q[4];
ry(1.1893364209493855) q[5];
cx q[4],q[5];
ry(-2.6993290209587064) q[4];
ry(-0.15910972297964676) q[5];
cx q[4],q[5];
ry(2.9114223674168453) q[5];
ry(-1.6682040841822534) q[6];
cx q[5],q[6];
ry(-2.1798498164121574) q[5];
ry(1.0027494639509698) q[6];
cx q[5],q[6];
ry(-0.0992364221464932) q[6];
ry(-1.0239318222573264) q[7];
cx q[6],q[7];
ry(1.4283744824374072) q[6];
ry(2.1359851411502193) q[7];
cx q[6],q[7];
ry(-2.0526429063348095) q[0];
ry(0.13886408249064738) q[1];
cx q[0],q[1];
ry(2.033386495213306) q[0];
ry(2.852246126961324) q[1];
cx q[0],q[1];
ry(2.331011970324631) q[1];
ry(1.1164055798928043) q[2];
cx q[1],q[2];
ry(-1.3569809609837096) q[1];
ry(-2.8447098301455243) q[2];
cx q[1],q[2];
ry(-3.0366469805537033) q[2];
ry(2.190120035707751) q[3];
cx q[2],q[3];
ry(-1.9751488742693835) q[2];
ry(1.9239818198518195) q[3];
cx q[2],q[3];
ry(1.425302406997168) q[3];
ry(2.1021708796789245) q[4];
cx q[3],q[4];
ry(0.13140981119489936) q[3];
ry(1.967601645907216) q[4];
cx q[3],q[4];
ry(-1.2903363389511417) q[4];
ry(1.3641948518751068) q[5];
cx q[4],q[5];
ry(-1.126049368308535) q[4];
ry(-1.571161769050558) q[5];
cx q[4],q[5];
ry(2.339743532889764) q[5];
ry(-2.977459344025914) q[6];
cx q[5],q[6];
ry(-2.3237157508091966) q[5];
ry(2.3849347328757013) q[6];
cx q[5],q[6];
ry(-0.32364844610929455) q[6];
ry(1.421118640420202) q[7];
cx q[6],q[7];
ry(2.8880782479131475) q[6];
ry(2.3323851127220583) q[7];
cx q[6],q[7];
ry(0.6259169722286239) q[0];
ry(1.4965805434335282) q[1];
cx q[0],q[1];
ry(1.7659669809798708) q[0];
ry(-3.036410546677526) q[1];
cx q[0],q[1];
ry(-3.027445231853718) q[1];
ry(-1.1997800756211483) q[2];
cx q[1],q[2];
ry(-2.369365262401725) q[1];
ry(1.8039999326725251) q[2];
cx q[1],q[2];
ry(2.732304797529488) q[2];
ry(-0.38924671108780373) q[3];
cx q[2],q[3];
ry(-0.7775594623042008) q[2];
ry(1.0150307373694485) q[3];
cx q[2],q[3];
ry(2.5557592682522947) q[3];
ry(-0.02072610028435154) q[4];
cx q[3],q[4];
ry(-0.5309998779319963) q[3];
ry(0.08063271453714727) q[4];
cx q[3],q[4];
ry(1.3608720861528294) q[4];
ry(2.23596613190453) q[5];
cx q[4],q[5];
ry(0.7843956295092799) q[4];
ry(-2.0665141823328765) q[5];
cx q[4],q[5];
ry(2.122150951189255) q[5];
ry(-1.2036882410973613) q[6];
cx q[5],q[6];
ry(2.1670725139625464) q[5];
ry(0.17001747818846535) q[6];
cx q[5],q[6];
ry(0.015696378352434922) q[6];
ry(1.613837938550529) q[7];
cx q[6],q[7];
ry(2.2931448054605337) q[6];
ry(2.1684401501545008) q[7];
cx q[6],q[7];
ry(2.623091890668642) q[0];
ry(2.257308994506215) q[1];
cx q[0],q[1];
ry(-2.6716297937388567) q[0];
ry(2.613158286134018) q[1];
cx q[0],q[1];
ry(-0.9348658200787436) q[1];
ry(-0.9486863045063259) q[2];
cx q[1],q[2];
ry(0.07546181977206423) q[1];
ry(-2.453699036257882) q[2];
cx q[1],q[2];
ry(-1.6447418892473884) q[2];
ry(0.17916920729520158) q[3];
cx q[2],q[3];
ry(1.4561098996642878) q[2];
ry(0.8446523604964025) q[3];
cx q[2],q[3];
ry(-0.14044844726196715) q[3];
ry(-0.5927358479392026) q[4];
cx q[3],q[4];
ry(3.11192319736892) q[3];
ry(-0.410065839198535) q[4];
cx q[3],q[4];
ry(1.0049253114083492) q[4];
ry(-0.6035243369200156) q[5];
cx q[4],q[5];
ry(0.7045590393190766) q[4];
ry(-0.44315328850410174) q[5];
cx q[4],q[5];
ry(0.06261132446022975) q[5];
ry(0.7144371279312675) q[6];
cx q[5],q[6];
ry(1.0095324182715326) q[5];
ry(0.9897720926606627) q[6];
cx q[5],q[6];
ry(2.915483072926592) q[6];
ry(-1.763169840684963) q[7];
cx q[6],q[7];
ry(-1.4195542480454741) q[6];
ry(1.2941406461590408) q[7];
cx q[6],q[7];
ry(0.7175350015594847) q[0];
ry(-1.6745733402138248) q[1];
cx q[0],q[1];
ry(-0.04855232769862994) q[0];
ry(1.5345007892668263) q[1];
cx q[0],q[1];
ry(2.0518803357428883) q[1];
ry(-2.5381291709651856) q[2];
cx q[1],q[2];
ry(0.6847001466693987) q[1];
ry(-0.19026944643852683) q[2];
cx q[1],q[2];
ry(0.5707489689823252) q[2];
ry(1.1152430202213832) q[3];
cx q[2],q[3];
ry(0.8484898105475197) q[2];
ry(-1.7634083236481048) q[3];
cx q[2],q[3];
ry(1.914795443480136) q[3];
ry(1.7925294635908111) q[4];
cx q[3],q[4];
ry(1.8554051470087467) q[3];
ry(0.016769101258145902) q[4];
cx q[3],q[4];
ry(-2.142332602847562) q[4];
ry(-1.031566776321573) q[5];
cx q[4],q[5];
ry(0.5620584972060838) q[4];
ry(-0.17415477176891883) q[5];
cx q[4],q[5];
ry(-0.5381685841656473) q[5];
ry(-0.699230418742645) q[6];
cx q[5],q[6];
ry(1.575547714360133) q[5];
ry(2.396964025386273) q[6];
cx q[5],q[6];
ry(-2.6847056187500677) q[6];
ry(-2.6311830783870915) q[7];
cx q[6],q[7];
ry(1.791499693422647) q[6];
ry(2.727636974535255) q[7];
cx q[6],q[7];
ry(-0.014955401717797301) q[0];
ry(-1.803211433452911) q[1];
cx q[0],q[1];
ry(-1.82887144788425) q[0];
ry(-1.4689623246634844) q[1];
cx q[0],q[1];
ry(1.2917810813326875) q[1];
ry(-2.3119310668601445) q[2];
cx q[1],q[2];
ry(-2.3029988431498754) q[1];
ry(-2.846355320849585) q[2];
cx q[1],q[2];
ry(-1.107049236348077) q[2];
ry(-2.5046348906276665) q[3];
cx q[2],q[3];
ry(-2.918616787172538) q[2];
ry(2.358523858580654) q[3];
cx q[2],q[3];
ry(2.693250782162549) q[3];
ry(0.5462404513285399) q[4];
cx q[3],q[4];
ry(1.7414830230203249) q[3];
ry(0.21479927702266696) q[4];
cx q[3],q[4];
ry(-0.698615797645855) q[4];
ry(-0.2535583706480695) q[5];
cx q[4],q[5];
ry(-0.8666191324363606) q[4];
ry(2.4126289917155517) q[5];
cx q[4],q[5];
ry(-0.325471238728759) q[5];
ry(-2.942393815079203) q[6];
cx q[5],q[6];
ry(1.1271367045040694) q[5];
ry(2.0587156032571574) q[6];
cx q[5],q[6];
ry(-2.14079818301636) q[6];
ry(1.7259566888061897) q[7];
cx q[6],q[7];
ry(0.925152846532149) q[6];
ry(-1.4565797153270912) q[7];
cx q[6],q[7];
ry(-1.0036668945878406) q[0];
ry(1.6831997314133451) q[1];
ry(-2.0158793867156) q[2];
ry(0.5509772612785975) q[3];
ry(0.6152361226467523) q[4];
ry(3.1137942607879148) q[5];
ry(2.5589112928845372) q[6];
ry(-1.0897072870985411) q[7];