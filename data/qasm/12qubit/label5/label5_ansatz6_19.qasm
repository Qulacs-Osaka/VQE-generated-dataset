OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.0655273491918171) q[0];
ry(2.3889087618558005) q[1];
cx q[0],q[1];
ry(2.615474909375169) q[0];
ry(-1.7114335398128757) q[1];
cx q[0],q[1];
ry(0.5795078186800487) q[1];
ry(-2.934353836685145) q[2];
cx q[1],q[2];
ry(-0.8440709365862675) q[1];
ry(-2.744290009779726) q[2];
cx q[1],q[2];
ry(-1.5576051781979618) q[2];
ry(3.1049642654852905) q[3];
cx q[2],q[3];
ry(-2.4686131608478314) q[2];
ry(2.8074470982248325) q[3];
cx q[2],q[3];
ry(-0.17360218316915393) q[3];
ry(-1.520930251493211) q[4];
cx q[3],q[4];
ry(0.09712106420856446) q[3];
ry(-0.08756410046936125) q[4];
cx q[3],q[4];
ry(-0.36894976019909365) q[4];
ry(2.429611869814533) q[5];
cx q[4],q[5];
ry(1.6414183643427145) q[4];
ry(-1.5666762327597632) q[5];
cx q[4],q[5];
ry(-0.20479969150484578) q[5];
ry(0.318747908049011) q[6];
cx q[5],q[6];
ry(2.7007424544843275) q[5];
ry(-2.918947866306444) q[6];
cx q[5],q[6];
ry(-0.785042509224141) q[6];
ry(-2.3910773021523077) q[7];
cx q[6],q[7];
ry(-2.233616860986636) q[6];
ry(0.39676740528150867) q[7];
cx q[6],q[7];
ry(-1.4691270795524418) q[7];
ry(2.4492507654296323) q[8];
cx q[7],q[8];
ry(0.15908534019549947) q[7];
ry(-3.104701260855223) q[8];
cx q[7],q[8];
ry(-2.2676677617619267) q[8];
ry(0.15522633369548977) q[9];
cx q[8],q[9];
ry(1.890843667372514) q[8];
ry(-1.270527041427255) q[9];
cx q[8],q[9];
ry(-1.207721294426685) q[9];
ry(2.8558387201501714) q[10];
cx q[9],q[10];
ry(-2.9099140166338517) q[9];
ry(-0.48034889835190386) q[10];
cx q[9],q[10];
ry(-1.443301402275961) q[10];
ry(-1.3635456099940568) q[11];
cx q[10],q[11];
ry(1.6138012325611957) q[10];
ry(2.5313728026905458) q[11];
cx q[10],q[11];
ry(2.814643025136592) q[0];
ry(0.573403534198488) q[1];
cx q[0],q[1];
ry(-0.3886724310014907) q[0];
ry(1.52100212564529) q[1];
cx q[0],q[1];
ry(-0.9405642943510825) q[1];
ry(-2.8790566643433113) q[2];
cx q[1],q[2];
ry(2.5092253879303223) q[1];
ry(2.7090470663396773) q[2];
cx q[1],q[2];
ry(0.7918037048432343) q[2];
ry(-2.463104748822018) q[3];
cx q[2],q[3];
ry(2.920312613159349) q[2];
ry(-1.4607539754788315) q[3];
cx q[2],q[3];
ry(0.16753171672903822) q[3];
ry(-2.3070536856378556) q[4];
cx q[3],q[4];
ry(-1.478397857197276) q[3];
ry(0.14315069181150653) q[4];
cx q[3],q[4];
ry(-3.133358335598097) q[4];
ry(2.049397172526165) q[5];
cx q[4],q[5];
ry(-2.863679139281697) q[4];
ry(-2.6428269607411656) q[5];
cx q[4],q[5];
ry(-0.7907753252377407) q[5];
ry(-1.5158571059867922) q[6];
cx q[5],q[6];
ry(-2.7284121667099797) q[5];
ry(-1.1530516097295083) q[6];
cx q[5],q[6];
ry(-2.753052343543623) q[6];
ry(2.3922342586216545) q[7];
cx q[6],q[7];
ry(1.3699507717822375) q[6];
ry(3.044173686560326) q[7];
cx q[6],q[7];
ry(-1.8822316921029332) q[7];
ry(-2.8563434588080776) q[8];
cx q[7],q[8];
ry(-1.447623077521869) q[7];
ry(-0.8254601275093103) q[8];
cx q[7],q[8];
ry(-0.511742682243691) q[8];
ry(-1.8128362918671677) q[9];
cx q[8],q[9];
ry(-2.11652944243947) q[8];
ry(1.24689730108667) q[9];
cx q[8],q[9];
ry(-2.5821543061074625) q[9];
ry(0.28415334528876246) q[10];
cx q[9],q[10];
ry(-1.0967376274249163) q[9];
ry(0.22612236848734657) q[10];
cx q[9],q[10];
ry(3.1377020330552177) q[10];
ry(1.8822086258998318) q[11];
cx q[10],q[11];
ry(0.21705048747454464) q[10];
ry(0.6680426850337994) q[11];
cx q[10],q[11];
ry(1.0617714862470802) q[0];
ry(2.4919926057265096) q[1];
cx q[0],q[1];
ry(0.5524614576691771) q[0];
ry(-2.796109478769818) q[1];
cx q[0],q[1];
ry(2.26428423086167) q[1];
ry(-2.1698763604817923) q[2];
cx q[1],q[2];
ry(-0.06976098865198178) q[1];
ry(-0.46347061838742576) q[2];
cx q[1],q[2];
ry(2.3623711927521533) q[2];
ry(-0.7451063612612062) q[3];
cx q[2],q[3];
ry(0.70251758415289) q[2];
ry(-0.21074933063055215) q[3];
cx q[2],q[3];
ry(-2.7078068510053557) q[3];
ry(-2.158051064234316) q[4];
cx q[3],q[4];
ry(0.2256348149780747) q[3];
ry(3.054159187699612) q[4];
cx q[3],q[4];
ry(0.02939627484501184) q[4];
ry(-2.981843096423873) q[5];
cx q[4],q[5];
ry(2.1639530911403257) q[4];
ry(-1.1641799909758392) q[5];
cx q[4],q[5];
ry(-2.4900943287670585) q[5];
ry(0.6265628324910576) q[6];
cx q[5],q[6];
ry(-2.9121998963657494) q[5];
ry(-2.604328775647217) q[6];
cx q[5],q[6];
ry(-2.633420670056408) q[6];
ry(-0.9864278311812492) q[7];
cx q[6],q[7];
ry(-0.05647266502047987) q[6];
ry(2.9816053652799885) q[7];
cx q[6],q[7];
ry(-2.263378771966665) q[7];
ry(-1.196048798659509) q[8];
cx q[7],q[8];
ry(0.2808070669540163) q[7];
ry(-1.6041937716568841) q[8];
cx q[7],q[8];
ry(-2.343746148473127) q[8];
ry(3.1024638197111067) q[9];
cx q[8],q[9];
ry(1.0318705678133098) q[8];
ry(-0.3051637275576725) q[9];
cx q[8],q[9];
ry(-2.772601086065793) q[9];
ry(-2.9365101500595046) q[10];
cx q[9],q[10];
ry(-2.2087363701881353) q[9];
ry(1.0274512863125258) q[10];
cx q[9],q[10];
ry(-1.9880188588765257) q[10];
ry(2.020363609724143) q[11];
cx q[10],q[11];
ry(1.7794845471450695) q[10];
ry(2.8433797410450103) q[11];
cx q[10],q[11];
ry(-0.4167386153549208) q[0];
ry(-1.1771413642280848) q[1];
cx q[0],q[1];
ry(2.992272131069166) q[0];
ry(-2.228427670448755) q[1];
cx q[0],q[1];
ry(-0.9161436022532365) q[1];
ry(1.4973484807007074) q[2];
cx q[1],q[2];
ry(-3.0214326894883263) q[1];
ry(1.532156672352742) q[2];
cx q[1],q[2];
ry(-2.48219346205097) q[2];
ry(-2.5157089610179555) q[3];
cx q[2],q[3];
ry(-3.0453459939681644) q[2];
ry(-0.905213501199773) q[3];
cx q[2],q[3];
ry(-0.6847082815239759) q[3];
ry(-0.6425194985024067) q[4];
cx q[3],q[4];
ry(-0.7678102614899741) q[3];
ry(-2.696619507216786) q[4];
cx q[3],q[4];
ry(2.3196710224253816) q[4];
ry(-0.06852996558104861) q[5];
cx q[4],q[5];
ry(-0.20976059437203123) q[4];
ry(-2.322754697233294) q[5];
cx q[4],q[5];
ry(-0.8677746320838322) q[5];
ry(-2.552715597818761) q[6];
cx q[5],q[6];
ry(-2.1420770621065053) q[5];
ry(0.08451708478641198) q[6];
cx q[5],q[6];
ry(1.8650663871595183) q[6];
ry(-2.718206895611937) q[7];
cx q[6],q[7];
ry(-3.101419532224069) q[6];
ry(2.4474814463307553) q[7];
cx q[6],q[7];
ry(1.7656761008868918) q[7];
ry(1.893655702654284) q[8];
cx q[7],q[8];
ry(2.84089510327545) q[7];
ry(-3.0955441760066136) q[8];
cx q[7],q[8];
ry(3.062464449451336) q[8];
ry(1.2766908256050618) q[9];
cx q[8],q[9];
ry(3.072366744141229) q[8];
ry(1.0641670312781835) q[9];
cx q[8],q[9];
ry(-0.35906852191707905) q[9];
ry(-1.8842414209384941) q[10];
cx q[9],q[10];
ry(-2.163218898888911) q[9];
ry(-1.9031977134163902) q[10];
cx q[9],q[10];
ry(1.8961317933523987) q[10];
ry(2.7087692784973387) q[11];
cx q[10],q[11];
ry(2.840509652253279) q[10];
ry(1.3292983685886428) q[11];
cx q[10],q[11];
ry(0.03724734395402507) q[0];
ry(-1.6217338552701461) q[1];
cx q[0],q[1];
ry(-0.9800699810541811) q[0];
ry(0.89526421630439) q[1];
cx q[0],q[1];
ry(2.8989093093134) q[1];
ry(-1.784847833745804) q[2];
cx q[1],q[2];
ry(-1.171581106407448) q[1];
ry(-2.3488911680674938) q[2];
cx q[1],q[2];
ry(-2.73086022719623) q[2];
ry(-2.9846061441038203) q[3];
cx q[2],q[3];
ry(0.6442397383549263) q[2];
ry(-2.5842496647451956) q[3];
cx q[2],q[3];
ry(-0.11889262435666303) q[3];
ry(-1.2617927563234952) q[4];
cx q[3],q[4];
ry(-1.6225616562052334) q[3];
ry(-2.0305718006927096) q[4];
cx q[3],q[4];
ry(-2.9455782007926277) q[4];
ry(0.05520452466334901) q[5];
cx q[4],q[5];
ry(-3.050836432673357) q[4];
ry(2.785267389576274) q[5];
cx q[4],q[5];
ry(-1.3085651344005214) q[5];
ry(1.6014715475477703) q[6];
cx q[5],q[6];
ry(-2.0540853684794245) q[5];
ry(-3.1276566171286575) q[6];
cx q[5],q[6];
ry(1.3908622131957171) q[6];
ry(-0.0533691019281724) q[7];
cx q[6],q[7];
ry(-0.1445707418959472) q[6];
ry(-0.7139218555526643) q[7];
cx q[6],q[7];
ry(-1.8276836601586848) q[7];
ry(-0.50971280799268) q[8];
cx q[7],q[8];
ry(-2.7067083723246523) q[7];
ry(-1.4054060677704046) q[8];
cx q[7],q[8];
ry(2.356510913139218) q[8];
ry(-1.223430083827341) q[9];
cx q[8],q[9];
ry(-0.6851144527535658) q[8];
ry(2.0899082594692375) q[9];
cx q[8],q[9];
ry(-0.20645055727592407) q[9];
ry(0.880951730717681) q[10];
cx q[9],q[10];
ry(1.7274331105348306) q[9];
ry(1.3715123131998714) q[10];
cx q[9],q[10];
ry(2.9669787254921776) q[10];
ry(2.1507693646178483) q[11];
cx q[10],q[11];
ry(-1.451093251606099) q[10];
ry(2.07809014943879) q[11];
cx q[10],q[11];
ry(0.43966755148138326) q[0];
ry(-1.9263951392045984) q[1];
cx q[0],q[1];
ry(0.050245307510647486) q[0];
ry(1.7123995246591788) q[1];
cx q[0],q[1];
ry(-1.026208633684755) q[1];
ry(-0.003386692666335733) q[2];
cx q[1],q[2];
ry(-0.5770883950028246) q[1];
ry(0.08587681379807648) q[2];
cx q[1],q[2];
ry(2.204564538557119) q[2];
ry(-0.16005014105722193) q[3];
cx q[2],q[3];
ry(0.609222848315024) q[2];
ry(0.15255426180863338) q[3];
cx q[2],q[3];
ry(-2.6239096931981263) q[3];
ry(0.03268546913062369) q[4];
cx q[3],q[4];
ry(-2.9115564769394777) q[3];
ry(1.8545765212962504) q[4];
cx q[3],q[4];
ry(2.2325769873832018) q[4];
ry(2.8821243313573) q[5];
cx q[4],q[5];
ry(1.975891428241324) q[4];
ry(0.6905152880587195) q[5];
cx q[4],q[5];
ry(-1.0170527501148048) q[5];
ry(-1.2524865987726224) q[6];
cx q[5],q[6];
ry(3.065823555358334) q[5];
ry(-2.911721682108781) q[6];
cx q[5],q[6];
ry(3.1354414098840984) q[6];
ry(-1.7803223889257629) q[7];
cx q[6],q[7];
ry(3.1400114617276786) q[6];
ry(-1.9343803872049337) q[7];
cx q[6],q[7];
ry(0.35337660685880934) q[7];
ry(3.0894694204755857) q[8];
cx q[7],q[8];
ry(1.8343054518488646) q[7];
ry(2.868449058704236) q[8];
cx q[7],q[8];
ry(-2.3642780894322017) q[8];
ry(-0.08893717593239715) q[9];
cx q[8],q[9];
ry(2.851601490844065) q[8];
ry(-3.091812080030784) q[9];
cx q[8],q[9];
ry(-1.2439543294476025) q[9];
ry(2.406434650002852) q[10];
cx q[9],q[10];
ry(-1.6965089400207458) q[9];
ry(-2.5981439694768382) q[10];
cx q[9],q[10];
ry(2.4870446356814218) q[10];
ry(-0.22537971970988746) q[11];
cx q[10],q[11];
ry(2.8767630321261453) q[10];
ry(2.1477379013068116) q[11];
cx q[10],q[11];
ry(-3.046878313513867) q[0];
ry(-2.056406778844898) q[1];
cx q[0],q[1];
ry(3.0621097961626114) q[0];
ry(1.7205701566008178) q[1];
cx q[0],q[1];
ry(-0.40248527349897856) q[1];
ry(0.6596361386153173) q[2];
cx q[1],q[2];
ry(2.263488135271634) q[1];
ry(0.7941830076669882) q[2];
cx q[1],q[2];
ry(-1.3269327850618005) q[2];
ry(-1.8975867533366673) q[3];
cx q[2],q[3];
ry(2.4869947121106266) q[2];
ry(2.5472069793004684) q[3];
cx q[2],q[3];
ry(2.737744975797886) q[3];
ry(1.8174790068532163) q[4];
cx q[3],q[4];
ry(-0.5907747662209824) q[3];
ry(2.287689598698369) q[4];
cx q[3],q[4];
ry(-0.37410728236459495) q[4];
ry(-0.4456408222794663) q[5];
cx q[4],q[5];
ry(-2.45083243509918) q[4];
ry(1.862874008406827) q[5];
cx q[4],q[5];
ry(2.379860795023529) q[5];
ry(-0.7654233858568256) q[6];
cx q[5],q[6];
ry(2.5156804170865077) q[5];
ry(2.5458294394084073) q[6];
cx q[5],q[6];
ry(0.8809145139333969) q[6];
ry(-0.14705161091698163) q[7];
cx q[6],q[7];
ry(-2.841135139336678) q[6];
ry(-0.008820430053420333) q[7];
cx q[6],q[7];
ry(1.2217711622539928) q[7];
ry(1.0762028084538837) q[8];
cx q[7],q[8];
ry(2.353984806664951) q[7];
ry(-2.8689003228409975) q[8];
cx q[7],q[8];
ry(1.080047413038483) q[8];
ry(-0.2524548079971609) q[9];
cx q[8],q[9];
ry(1.5511146056342282) q[8];
ry(2.1649541940231463) q[9];
cx q[8],q[9];
ry(-2.9035123747652976) q[9];
ry(-2.502496591243603) q[10];
cx q[9],q[10];
ry(2.1242785963149946) q[9];
ry(1.2631336941468785) q[10];
cx q[9],q[10];
ry(1.0461590529928089) q[10];
ry(-1.5739642714613948) q[11];
cx q[10],q[11];
ry(0.8855845883547124) q[10];
ry(-1.6762439354858387) q[11];
cx q[10],q[11];
ry(-2.5661982493284397) q[0];
ry(-0.48309812550326825) q[1];
cx q[0],q[1];
ry(0.9179005619281595) q[0];
ry(0.12732841062353256) q[1];
cx q[0],q[1];
ry(-1.5312487053450434) q[1];
ry(-2.91036689306436) q[2];
cx q[1],q[2];
ry(-3.1092053355361937) q[1];
ry(2.7097031570600305) q[2];
cx q[1],q[2];
ry(-3.003275505352529) q[2];
ry(2.9885107466367673) q[3];
cx q[2],q[3];
ry(-2.172701847273398) q[2];
ry(2.6291580798393714) q[3];
cx q[2],q[3];
ry(-2.327690826113427) q[3];
ry(0.8710467365021978) q[4];
cx q[3],q[4];
ry(1.3387794069355199) q[3];
ry(2.667659420972293) q[4];
cx q[3],q[4];
ry(-2.3716924902889978) q[4];
ry(-2.7588248926874894) q[5];
cx q[4],q[5];
ry(-0.33323699872376267) q[4];
ry(2.030996559222899) q[5];
cx q[4],q[5];
ry(2.012447181587314) q[5];
ry(1.5869983889735924) q[6];
cx q[5],q[6];
ry(0.051317968663853186) q[5];
ry(1.507208976922017) q[6];
cx q[5],q[6];
ry(1.6894372532058686) q[6];
ry(0.9720819959821725) q[7];
cx q[6],q[7];
ry(2.945380850982753) q[6];
ry(0.015513320806101313) q[7];
cx q[6],q[7];
ry(-1.6928532972309756) q[7];
ry(1.2238922100558218) q[8];
cx q[7],q[8];
ry(-3.062573320378005) q[7];
ry(1.0807569837452287) q[8];
cx q[7],q[8];
ry(1.8940438675489633) q[8];
ry(-2.6133894404681643) q[9];
cx q[8],q[9];
ry(0.6538077425068625) q[8];
ry(0.24425624646282085) q[9];
cx q[8],q[9];
ry(-0.11747998497425714) q[9];
ry(-1.2585270300068503) q[10];
cx q[9],q[10];
ry(0.43278372776434537) q[9];
ry(-1.0408765764421952) q[10];
cx q[9],q[10];
ry(0.48008054217209667) q[10];
ry(-0.02859580093551095) q[11];
cx q[10],q[11];
ry(-0.04153684508522559) q[10];
ry(-0.12198264089253819) q[11];
cx q[10],q[11];
ry(3.0446764377096245) q[0];
ry(-0.3510372818318581) q[1];
cx q[0],q[1];
ry(1.4634288000839648) q[0];
ry(-1.6281490900285371) q[1];
cx q[0],q[1];
ry(-1.0808091475686332) q[1];
ry(-1.7698032322822537) q[2];
cx q[1],q[2];
ry(3.028325048591656) q[1];
ry(-0.8538605271550885) q[2];
cx q[1],q[2];
ry(-0.6107645009288869) q[2];
ry(2.057298116652314) q[3];
cx q[2],q[3];
ry(-0.5759010345133877) q[2];
ry(0.8962235777042887) q[3];
cx q[2],q[3];
ry(-1.989719601600687) q[3];
ry(2.525474392206677) q[4];
cx q[3],q[4];
ry(3.1353889808391466) q[3];
ry(1.1710982912063885) q[4];
cx q[3],q[4];
ry(-0.7976376908866154) q[4];
ry(-1.7805746816014485) q[5];
cx q[4],q[5];
ry(0.9454544091652903) q[4];
ry(-0.3733741399818196) q[5];
cx q[4],q[5];
ry(-1.4733144566743306) q[5];
ry(1.717007631326056) q[6];
cx q[5],q[6];
ry(-0.1776784865804677) q[5];
ry(-1.0118602319678525) q[6];
cx q[5],q[6];
ry(-1.0620287655880762) q[6];
ry(-1.5002038810444356) q[7];
cx q[6],q[7];
ry(2.1900846900537743) q[6];
ry(-3.1092323508638327) q[7];
cx q[6],q[7];
ry(-1.1823431076742486) q[7];
ry(-1.3586783273111946) q[8];
cx q[7],q[8];
ry(1.141316808659345) q[7];
ry(1.987477959227461) q[8];
cx q[7],q[8];
ry(-2.0842555099855304) q[8];
ry(2.4334568046533387) q[9];
cx q[8],q[9];
ry(0.34855107781211525) q[8];
ry(-1.5752049936674855) q[9];
cx q[8],q[9];
ry(1.923373155047224) q[9];
ry(-3.083178056006954) q[10];
cx q[9],q[10];
ry(2.4008539230990653) q[9];
ry(1.0236270383282138) q[10];
cx q[9],q[10];
ry(-2.982116373031454) q[10];
ry(1.9246264579628847) q[11];
cx q[10],q[11];
ry(-1.4521997257393373) q[10];
ry(0.1907556252467964) q[11];
cx q[10],q[11];
ry(2.071650670575102) q[0];
ry(-0.8454934389931346) q[1];
cx q[0],q[1];
ry(2.6991142784894198) q[0];
ry(-0.8271735538361851) q[1];
cx q[0],q[1];
ry(0.003119784156498717) q[1];
ry(-2.5656900118794006) q[2];
cx q[1],q[2];
ry(-3.0406564297501926) q[1];
ry(0.009453483473540957) q[2];
cx q[1],q[2];
ry(2.8820878073668377) q[2];
ry(-1.4402314034054697) q[3];
cx q[2],q[3];
ry(2.698712042569691) q[2];
ry(1.877003135503955) q[3];
cx q[2],q[3];
ry(-2.933996462170733) q[3];
ry(1.4015064943836215) q[4];
cx q[3],q[4];
ry(3.0289234976821735) q[3];
ry(-1.8969816942647897) q[4];
cx q[3],q[4];
ry(2.5619114286359386) q[4];
ry(0.07138002450625938) q[5];
cx q[4],q[5];
ry(-1.523753358847439) q[4];
ry(-1.0816873932048203) q[5];
cx q[4],q[5];
ry(2.1674901242136597) q[5];
ry(1.0405313358187702) q[6];
cx q[5],q[6];
ry(-3.1371894817623365) q[5];
ry(0.7733021109816819) q[6];
cx q[5],q[6];
ry(-2.72311259243708) q[6];
ry(-0.5491921726636616) q[7];
cx q[6],q[7];
ry(-0.8081938788693618) q[6];
ry(0.05658021946974013) q[7];
cx q[6],q[7];
ry(-2.609142423782919) q[7];
ry(-1.0327083188657589) q[8];
cx q[7],q[8];
ry(-2.2388966835755353) q[7];
ry(1.362494047464991) q[8];
cx q[7],q[8];
ry(-3.115805756533892) q[8];
ry(3.0113753571346757) q[9];
cx q[8],q[9];
ry(-1.7615332723216728) q[8];
ry(2.9055717432734243) q[9];
cx q[8],q[9];
ry(1.589928464738434) q[9];
ry(2.122866656704447) q[10];
cx q[9],q[10];
ry(1.469901741854529) q[9];
ry(1.712479464922349) q[10];
cx q[9],q[10];
ry(-1.7345765450198274) q[10];
ry(2.34097173195196) q[11];
cx q[10],q[11];
ry(2.029994957042975) q[10];
ry(-0.26067206484171734) q[11];
cx q[10],q[11];
ry(2.5381031166570622) q[0];
ry(-0.9546804888202657) q[1];
cx q[0],q[1];
ry(0.9187052576920136) q[0];
ry(3.0574935173042563) q[1];
cx q[0],q[1];
ry(-2.2614787432273813) q[1];
ry(1.3447908013599619) q[2];
cx q[1],q[2];
ry(-0.38051461895849714) q[1];
ry(-0.8736711302368255) q[2];
cx q[1],q[2];
ry(1.3477352833115157) q[2];
ry(-2.72638221932537) q[3];
cx q[2],q[3];
ry(-1.4576063646606423) q[2];
ry(-3.1343469782926974) q[3];
cx q[2],q[3];
ry(-0.8625314686168859) q[3];
ry(-1.7617326348525217) q[4];
cx q[3],q[4];
ry(-0.08729601156309703) q[3];
ry(-2.4769581715997724) q[4];
cx q[3],q[4];
ry(2.559033713885078) q[4];
ry(1.6457410204121556) q[5];
cx q[4],q[5];
ry(-1.3927948647999164) q[4];
ry(-0.6795236158396649) q[5];
cx q[4],q[5];
ry(2.6382689112277444) q[5];
ry(-1.7221635651341336) q[6];
cx q[5],q[6];
ry(3.1200062455030686) q[5];
ry(-1.3351309340636057) q[6];
cx q[5],q[6];
ry(-0.8633370505218786) q[6];
ry(1.0648967866575099) q[7];
cx q[6],q[7];
ry(0.45673614075538715) q[6];
ry(-3.1324295948704757) q[7];
cx q[6],q[7];
ry(-1.2461264842465904) q[7];
ry(-3.1033673257759786) q[8];
cx q[7],q[8];
ry(-2.1979268207469147) q[7];
ry(-2.0686611279994027) q[8];
cx q[7],q[8];
ry(0.8989952348557564) q[8];
ry(0.5436580413563771) q[9];
cx q[8],q[9];
ry(-1.9500353136172377) q[8];
ry(2.567259053438248) q[9];
cx q[8],q[9];
ry(-2.5060513319233078) q[9];
ry(-2.9485007865021533) q[10];
cx q[9],q[10];
ry(-0.9032462748995576) q[9];
ry(1.1583262837578614) q[10];
cx q[9],q[10];
ry(0.7717264105901681) q[10];
ry(-2.1983280650344934) q[11];
cx q[10],q[11];
ry(-3.1051545325625227) q[10];
ry(-2.3295699884476897) q[11];
cx q[10],q[11];
ry(2.7465495653179897) q[0];
ry(1.0837091271412271) q[1];
cx q[0],q[1];
ry(1.9536004983166164) q[0];
ry(2.775426284333017) q[1];
cx q[0],q[1];
ry(0.5268056349015744) q[1];
ry(1.0290799880946224) q[2];
cx q[1],q[2];
ry(-0.06780525576912932) q[1];
ry(-0.7934955033223501) q[2];
cx q[1],q[2];
ry(2.7338335672248886) q[2];
ry(-2.1867065940671857) q[3];
cx q[2],q[3];
ry(0.6719838468422911) q[2];
ry(1.8945202097232716) q[3];
cx q[2],q[3];
ry(-1.1490158480219346) q[3];
ry(2.018182585099826) q[4];
cx q[3],q[4];
ry(-0.5406181825049723) q[3];
ry(0.709149916508288) q[4];
cx q[3],q[4];
ry(0.8637326088489414) q[4];
ry(-2.6558748658725926) q[5];
cx q[4],q[5];
ry(0.9561586456622929) q[4];
ry(-3.0566718364122156) q[5];
cx q[4],q[5];
ry(-2.5886206701172223) q[5];
ry(-1.6209581337009933) q[6];
cx q[5],q[6];
ry(-0.09491111979360589) q[5];
ry(-0.9192785636076406) q[6];
cx q[5],q[6];
ry(2.585650386541137) q[6];
ry(0.6877725413118914) q[7];
cx q[6],q[7];
ry(0.10154037882854006) q[6];
ry(0.00011041249111531926) q[7];
cx q[6],q[7];
ry(1.1664193250605548) q[7];
ry(-2.580830736263462) q[8];
cx q[7],q[8];
ry(2.4025170515900265) q[7];
ry(0.5782985927827509) q[8];
cx q[7],q[8];
ry(-2.372634290715301) q[8];
ry(2.0846500684248843) q[9];
cx q[8],q[9];
ry(-0.7064674735590066) q[8];
ry(-0.9292557377996982) q[9];
cx q[8],q[9];
ry(-0.5488785813896919) q[9];
ry(-2.239020239418698) q[10];
cx q[9],q[10];
ry(-2.3120312454306107) q[9];
ry(-0.09046833778050596) q[10];
cx q[9],q[10];
ry(0.035518654027014485) q[10];
ry(-1.2707760220783575) q[11];
cx q[10],q[11];
ry(1.242775994674143) q[10];
ry(2.694729428322195) q[11];
cx q[10],q[11];
ry(-0.712543891597381) q[0];
ry(2.792196259350861) q[1];
cx q[0],q[1];
ry(-1.4579203372934826) q[0];
ry(1.437315753619738) q[1];
cx q[0],q[1];
ry(1.1925541191976514) q[1];
ry(-0.40459324956232307) q[2];
cx q[1],q[2];
ry(3.0604255329760237) q[1];
ry(0.636004065794678) q[2];
cx q[1],q[2];
ry(0.22807681382746808) q[2];
ry(0.8725761323897112) q[3];
cx q[2],q[3];
ry(-0.08148659654113742) q[2];
ry(0.29930773312269854) q[3];
cx q[2],q[3];
ry(2.25966584017558) q[3];
ry(-2.5245286754992775) q[4];
cx q[3],q[4];
ry(1.9115265569469013) q[3];
ry(-0.74744957171787) q[4];
cx q[3],q[4];
ry(2.093157162170368) q[4];
ry(1.5219443749354173) q[5];
cx q[4],q[5];
ry(2.0229433916850583) q[4];
ry(-1.6381412932912038) q[5];
cx q[4],q[5];
ry(-3.0912138793704105) q[5];
ry(-2.4279851146930973) q[6];
cx q[5],q[6];
ry(3.0818434537660835) q[5];
ry(-1.3993262040185304) q[6];
cx q[5],q[6];
ry(1.0963852660871312) q[6];
ry(-2.730976152679909) q[7];
cx q[6],q[7];
ry(-0.8368143126020318) q[6];
ry(0.05037625311788626) q[7];
cx q[6],q[7];
ry(2.9182022949782676) q[7];
ry(1.7262274367627954) q[8];
cx q[7],q[8];
ry(-2.807224430136795) q[7];
ry(2.9815229517873902) q[8];
cx q[7],q[8];
ry(0.9724622714074453) q[8];
ry(0.6006857651323401) q[9];
cx q[8],q[9];
ry(2.5045022047668084) q[8];
ry(-1.9164665774747907) q[9];
cx q[8],q[9];
ry(2.6084851941650222) q[9];
ry(-0.7592775248691378) q[10];
cx q[9],q[10];
ry(0.42166255705323685) q[9];
ry(-1.1234509296845916) q[10];
cx q[9],q[10];
ry(-1.9462995795659828) q[10];
ry(-1.9350341535726276) q[11];
cx q[10],q[11];
ry(-2.700044942305036) q[10];
ry(-0.7455239481135623) q[11];
cx q[10],q[11];
ry(0.36039978307947) q[0];
ry(-1.339180346008148) q[1];
cx q[0],q[1];
ry(-0.0027881635544728667) q[0];
ry(0.6533286684866884) q[1];
cx q[0],q[1];
ry(1.1479962569731486) q[1];
ry(0.30060933180717075) q[2];
cx q[1],q[2];
ry(-1.7372057474736016) q[1];
ry(-2.0552668442380364) q[2];
cx q[1],q[2];
ry(1.8804996818127753) q[2];
ry(2.7353513257188697) q[3];
cx q[2],q[3];
ry(0.018545052306468142) q[2];
ry(0.5810691193694906) q[3];
cx q[2],q[3];
ry(1.5012565782134875) q[3];
ry(-1.906234864564218) q[4];
cx q[3],q[4];
ry(1.0189772239099693) q[3];
ry(2.727259370039943) q[4];
cx q[3],q[4];
ry(2.661403965416581) q[4];
ry(-2.313221590424433) q[5];
cx q[4],q[5];
ry(-0.027829569910301725) q[4];
ry(2.95194181329526) q[5];
cx q[4],q[5];
ry(-2.9117977477787655) q[5];
ry(-1.4189764506100522) q[6];
cx q[5],q[6];
ry(0.8070036689037571) q[5];
ry(0.02460905510907196) q[6];
cx q[5],q[6];
ry(1.2178615019721457) q[6];
ry(2.62191646067773) q[7];
cx q[6],q[7];
ry(-1.017083928890746) q[6];
ry(-3.0877760730570043) q[7];
cx q[6],q[7];
ry(1.5859870621186236) q[7];
ry(-0.20865107160745272) q[8];
cx q[7],q[8];
ry(-0.017039807721729087) q[7];
ry(1.6702731625853362) q[8];
cx q[7],q[8];
ry(-0.20795250158228365) q[8];
ry(2.4661909273784293) q[9];
cx q[8],q[9];
ry(2.0177590594059343) q[8];
ry(-0.46360949124872874) q[9];
cx q[8],q[9];
ry(-0.036151639518954504) q[9];
ry(3.0231043382993965) q[10];
cx q[9],q[10];
ry(-1.694746167148234) q[9];
ry(-2.3920321977363264) q[10];
cx q[9],q[10];
ry(1.7461963110811811) q[10];
ry(0.38743915606982804) q[11];
cx q[10],q[11];
ry(0.10834521780443397) q[10];
ry(1.9484352646430176) q[11];
cx q[10],q[11];
ry(2.7800807720561362) q[0];
ry(0.31340418382782786) q[1];
cx q[0],q[1];
ry(2.0126732295929584) q[0];
ry(-1.3627730620542164) q[1];
cx q[0],q[1];
ry(3.1329566026888918) q[1];
ry(-0.5607327126061028) q[2];
cx q[1],q[2];
ry(2.1401251009764106) q[1];
ry(-1.181310891841286) q[2];
cx q[1],q[2];
ry(-2.1838123982879614) q[2];
ry(2.1870072495016784) q[3];
cx q[2],q[3];
ry(-3.137110019843485) q[2];
ry(-0.7594003928575948) q[3];
cx q[2],q[3];
ry(1.8864983165957598) q[3];
ry(0.7711618499808397) q[4];
cx q[3],q[4];
ry(2.670113634708942) q[3];
ry(2.467525016693005) q[4];
cx q[3],q[4];
ry(-2.991513950209926) q[4];
ry(1.7630150757674388) q[5];
cx q[4],q[5];
ry(-1.5990938407768134) q[4];
ry(-0.16489250498696695) q[5];
cx q[4],q[5];
ry(3.0262679117814235) q[5];
ry(-2.438048656415107) q[6];
cx q[5],q[6];
ry(0.015584451865899638) q[5];
ry(2.09526152144457) q[6];
cx q[5],q[6];
ry(-2.237663706402075) q[6];
ry(-1.0306956092736916) q[7];
cx q[6],q[7];
ry(-2.639142786580625) q[6];
ry(-3.125093920744245) q[7];
cx q[6],q[7];
ry(-1.8036164377565533) q[7];
ry(-3.0254675986329373) q[8];
cx q[7],q[8];
ry(-0.008390976543089401) q[7];
ry(3.083074418879149) q[8];
cx q[7],q[8];
ry(0.4129697285121834) q[8];
ry(2.6645864366389125) q[9];
cx q[8],q[9];
ry(-1.4544779138108888) q[8];
ry(-0.8988259827542926) q[9];
cx q[8],q[9];
ry(-2.580840140025238) q[9];
ry(0.6312240023661706) q[10];
cx q[9],q[10];
ry(3.041475496932717) q[9];
ry(2.877318488242342) q[10];
cx q[9],q[10];
ry(1.364551106575922) q[10];
ry(2.1688305772167844) q[11];
cx q[10],q[11];
ry(-1.3945188797770305) q[10];
ry(-1.021493581665868) q[11];
cx q[10],q[11];
ry(0.6118663672609674) q[0];
ry(0.43845935914237444) q[1];
cx q[0],q[1];
ry(1.877494025805733) q[0];
ry(-1.3413973627487) q[1];
cx q[0],q[1];
ry(1.7400086125032104) q[1];
ry(-2.966886814676501) q[2];
cx q[1],q[2];
ry(2.5300775423310706) q[1];
ry(1.044744769306252) q[2];
cx q[1],q[2];
ry(0.24204528621547006) q[2];
ry(2.2605140238693577) q[3];
cx q[2],q[3];
ry(3.135455374401534) q[2];
ry(1.416212100343765) q[3];
cx q[2],q[3];
ry(0.26192519653550245) q[3];
ry(-0.21347569273127487) q[4];
cx q[3],q[4];
ry(1.5619247180257654) q[3];
ry(0.9248602700837348) q[4];
cx q[3],q[4];
ry(-1.0282514658159458) q[4];
ry(-1.7400357095967154) q[5];
cx q[4],q[5];
ry(-1.5968439870479316) q[4];
ry(-0.011183197335235207) q[5];
cx q[4],q[5];
ry(1.5939962397285241) q[5];
ry(-1.4838792908337666) q[6];
cx q[5],q[6];
ry(3.0182631392290857) q[5];
ry(2.989621587741359) q[6];
cx q[5],q[6];
ry(1.7080125848680694) q[6];
ry(-2.5221097657306477) q[7];
cx q[6],q[7];
ry(0.4667894680293057) q[6];
ry(-0.011447802033939247) q[7];
cx q[6],q[7];
ry(-1.7304199471886845) q[7];
ry(-2.155501356864014) q[8];
cx q[7],q[8];
ry(-2.907089411142638) q[7];
ry(-0.44178115806518115) q[8];
cx q[7],q[8];
ry(2.062470401398728) q[8];
ry(-2.6013024632665145) q[9];
cx q[8],q[9];
ry(2.975065054429132) q[8];
ry(0.3028119364195003) q[9];
cx q[8],q[9];
ry(1.6512158528291332) q[9];
ry(0.269776176196836) q[10];
cx q[9],q[10];
ry(-2.747252357066938) q[9];
ry(-0.9499975997814021) q[10];
cx q[9],q[10];
ry(0.478697513763014) q[10];
ry(1.8076487693249188) q[11];
cx q[10],q[11];
ry(0.5547977209653076) q[10];
ry(2.4234687895435187) q[11];
cx q[10],q[11];
ry(-2.6123924487261774) q[0];
ry(2.195437841840798) q[1];
cx q[0],q[1];
ry(2.5521755282789442) q[0];
ry(-1.1369944457378125) q[1];
cx q[0],q[1];
ry(1.464836230776128) q[1];
ry(1.9663701966260074) q[2];
cx q[1],q[2];
ry(-1.4330060447840796) q[1];
ry(-3.064942278816404) q[2];
cx q[1],q[2];
ry(-3.1102878450844984) q[2];
ry(1.8345957768545964) q[3];
cx q[2],q[3];
ry(0.30098572508742494) q[2];
ry(1.0955088302837652) q[3];
cx q[2],q[3];
ry(-3.0712604194227313) q[3];
ry(-2.723459489026735) q[4];
cx q[3],q[4];
ry(-3.102429048687909) q[3];
ry(-0.175365903281585) q[4];
cx q[3],q[4];
ry(-0.6536486865631019) q[4];
ry(-2.0772943223029463) q[5];
cx q[4],q[5];
ry(0.026781264461045777) q[4];
ry(-0.008819254322156212) q[5];
cx q[4],q[5];
ry(3.0819948657366143) q[5];
ry(-1.4962400232745603) q[6];
cx q[5],q[6];
ry(0.21637577846457923) q[5];
ry(1.0595703891641568) q[6];
cx q[5],q[6];
ry(-2.8919878873540146) q[6];
ry(2.4146870200604424) q[7];
cx q[6],q[7];
ry(3.0568249315081877) q[6];
ry(-3.1177359604429618) q[7];
cx q[6],q[7];
ry(0.16002106947119007) q[7];
ry(2.842707928832631) q[8];
cx q[7],q[8];
ry(-1.7868484250827317) q[7];
ry(-2.8012990071438444) q[8];
cx q[7],q[8];
ry(-1.7435235965472822) q[8];
ry(1.7200497591160628) q[9];
cx q[8],q[9];
ry(-3.074031229455047) q[8];
ry(0.21358822819747778) q[9];
cx q[8],q[9];
ry(-0.8463323155818552) q[9];
ry(2.626605104536778) q[10];
cx q[9],q[10];
ry(-1.6989820349428628) q[9];
ry(2.34038797076731) q[10];
cx q[9],q[10];
ry(1.6056631082587938) q[10];
ry(0.023184728799905407) q[11];
cx q[10],q[11];
ry(1.926238722519248) q[10];
ry(-0.811441843306783) q[11];
cx q[10],q[11];
ry(-2.944178852383373) q[0];
ry(-2.5209496164413316) q[1];
cx q[0],q[1];
ry(2.5146653977654) q[0];
ry(-2.6915753907866544) q[1];
cx q[0],q[1];
ry(2.4497844931925954) q[1];
ry(2.224856553275763) q[2];
cx q[1],q[2];
ry(-1.063294916628501) q[1];
ry(-3.0125990425596294) q[2];
cx q[1],q[2];
ry(-1.7430288973654395) q[2];
ry(3.1205956017927705) q[3];
cx q[2],q[3];
ry(-1.4193908454640465) q[2];
ry(2.2499347917666257) q[3];
cx q[2],q[3];
ry(1.547672978626351) q[3];
ry(1.5950891755162457) q[4];
cx q[3],q[4];
ry(-1.6286255967671333) q[3];
ry(-1.3358481546482626) q[4];
cx q[3],q[4];
ry(-1.5356075873376411) q[4];
ry(2.4479732191001116) q[5];
cx q[4],q[5];
ry(1.5326116249090784) q[4];
ry(1.4981424878712222) q[5];
cx q[4],q[5];
ry(1.4266245768924835) q[5];
ry(2.2291172585240053) q[6];
cx q[5],q[6];
ry(0.1732004843133519) q[5];
ry(-3.0927591389133906) q[6];
cx q[5],q[6];
ry(0.34403642642996196) q[6];
ry(0.9802504730791035) q[7];
cx q[6],q[7];
ry(3.1366065476227223) q[6];
ry(-0.20458803357822314) q[7];
cx q[6],q[7];
ry(2.5920809229430364) q[7];
ry(-1.835064617332259) q[8];
cx q[7],q[8];
ry(1.627590305814682) q[7];
ry(3.135207760559766) q[8];
cx q[7],q[8];
ry(0.8623999388262815) q[8];
ry(-1.978487335896804) q[9];
cx q[8],q[9];
ry(2.1685899417546066) q[8];
ry(2.165369687656109) q[9];
cx q[8],q[9];
ry(-1.7068815856562367) q[9];
ry(0.1840876997685328) q[10];
cx q[9],q[10];
ry(2.2076815153300178) q[9];
ry(0.7551585565454904) q[10];
cx q[9],q[10];
ry(0.49367486346277284) q[10];
ry(1.326521354016057) q[11];
cx q[10],q[11];
ry(1.6192395230617693) q[10];
ry(-0.0340474398957864) q[11];
cx q[10],q[11];
ry(0.8894780374793403) q[0];
ry(-3.131027102901411) q[1];
cx q[0],q[1];
ry(-1.3778638367736529) q[0];
ry(0.1870842161008911) q[1];
cx q[0],q[1];
ry(-0.27453398237929605) q[1];
ry(2.5618390352390477) q[2];
cx q[1],q[2];
ry(-1.5458798914397711) q[1];
ry(2.73863702439452) q[2];
cx q[1],q[2];
ry(-1.3320282984865301) q[2];
ry(-2.1491837799460454) q[3];
cx q[2],q[3];
ry(3.1157101029153975) q[2];
ry(1.6415590014551693) q[3];
cx q[2],q[3];
ry(0.7493772197689008) q[3];
ry(1.576457296723286) q[4];
cx q[3],q[4];
ry(1.3025187653656891) q[3];
ry(3.1187896257020724) q[4];
cx q[3],q[4];
ry(2.997987572185849) q[4];
ry(-1.2542541772006777) q[5];
cx q[4],q[5];
ry(-0.04441597754142561) q[4];
ry(0.017431522284608703) q[5];
cx q[4],q[5];
ry(-1.8241846514734692) q[5];
ry(1.5661331899208362) q[6];
cx q[5],q[6];
ry(-1.7472015201235591) q[5];
ry(0.008236557667907617) q[6];
cx q[5],q[6];
ry(-1.6020340134417823) q[6];
ry(-3.1078209583681975) q[7];
cx q[6],q[7];
ry(1.6533635212876927) q[6];
ry(-0.2532679285611782) q[7];
cx q[6],q[7];
ry(-0.4457344252829243) q[7];
ry(2.474105351572269) q[8];
cx q[7],q[8];
ry(-3.13906992221568) q[7];
ry(-0.00478464938831135) q[8];
cx q[7],q[8];
ry(1.7548598382142309) q[8];
ry(-1.051966174089216) q[9];
cx q[8],q[9];
ry(2.2477059456588684) q[8];
ry(2.448628620449889) q[9];
cx q[8],q[9];
ry(-1.4348981462127206) q[9];
ry(-0.17857124688926115) q[10];
cx q[9],q[10];
ry(-2.4079913227587197) q[9];
ry(1.6430271174360307) q[10];
cx q[9],q[10];
ry(2.3326287563508608) q[10];
ry(1.1149478641174613) q[11];
cx q[10],q[11];
ry(-0.0030752259909966498) q[10];
ry(-2.081243985151349) q[11];
cx q[10],q[11];
ry(-2.326879384849735) q[0];
ry(-0.6237065731827389) q[1];
cx q[0],q[1];
ry(1.5290903947155128) q[0];
ry(-1.5940990531128796) q[1];
cx q[0],q[1];
ry(2.1067452949690324) q[1];
ry(2.07186284526518) q[2];
cx q[1],q[2];
ry(-1.0494803575481584) q[1];
ry(0.24373700156917363) q[2];
cx q[1],q[2];
ry(-1.336115963111145) q[2];
ry(-3.1385318677293976) q[3];
cx q[2],q[3];
ry(0.05198898437456822) q[2];
ry(-0.4564808898651531) q[3];
cx q[2],q[3];
ry(0.5019695387430073) q[3];
ry(0.3383978930070216) q[4];
cx q[3],q[4];
ry(0.8859757031926521) q[3];
ry(3.1205499631207507) q[4];
cx q[3],q[4];
ry(1.4435426754149212) q[4];
ry(-2.125661488497305) q[5];
cx q[4],q[5];
ry(-0.041345513299855696) q[4];
ry(-0.0833449386000602) q[5];
cx q[4],q[5];
ry(0.8753934618090552) q[5];
ry(2.520733750229994) q[6];
cx q[5],q[6];
ry(-1.2577218938319672) q[5];
ry(1.8175257942228882) q[6];
cx q[5],q[6];
ry(-1.9488874097874551) q[6];
ry(-2.698807937608959) q[7];
cx q[6],q[7];
ry(-0.2669243501395059) q[6];
ry(-0.000843467497747606) q[7];
cx q[6],q[7];
ry(0.003524035133912597) q[7];
ry(-1.7121589884381836) q[8];
cx q[7],q[8];
ry(1.574493368682821) q[7];
ry(1.572477553808953) q[8];
cx q[7],q[8];
ry(3.1267753785478507) q[8];
ry(1.3236416575554966) q[9];
cx q[8],q[9];
ry(0.0012036292645145608) q[8];
ry(-0.15138051097073557) q[9];
cx q[8],q[9];
ry(1.3443582568274781) q[9];
ry(1.7111032985525774) q[10];
cx q[9],q[10];
ry(-2.4757631057261342) q[9];
ry(0.30392346249586133) q[10];
cx q[9],q[10];
ry(0.9241247509330011) q[10];
ry(-1.1036732248119066) q[11];
cx q[10],q[11];
ry(0.5674450591516654) q[10];
ry(1.5993437090079639) q[11];
cx q[10],q[11];
ry(0.8798850236977711) q[0];
ry(0.5252201242876167) q[1];
cx q[0],q[1];
ry(-2.9603006791604805) q[0];
ry(-0.6730424190324538) q[1];
cx q[0],q[1];
ry(-2.180446675192188) q[1];
ry(2.796294008520716) q[2];
cx q[1],q[2];
ry(-1.894703129587194) q[1];
ry(-0.07183000781926391) q[2];
cx q[1],q[2];
ry(-1.7889637355865604) q[2];
ry(-0.9590301748022971) q[3];
cx q[2],q[3];
ry(2.9797187267273912) q[2];
ry(1.1933168485382304) q[3];
cx q[2],q[3];
ry(-2.2271044542824665) q[3];
ry(-2.9949412490954646) q[4];
cx q[3],q[4];
ry(0.5888327096195886) q[3];
ry(-3.1381718291899223) q[4];
cx q[3],q[4];
ry(-1.187182381616817) q[4];
ry(-1.8515368601167776) q[5];
cx q[4],q[5];
ry(0.005551916216597776) q[4];
ry(0.002752452412241979) q[5];
cx q[4],q[5];
ry(-2.7853302502275143) q[5];
ry(1.8572515041318733) q[6];
cx q[5],q[6];
ry(-1.844879202694739) q[5];
ry(1.041049245987008) q[6];
cx q[5],q[6];
ry(-2.099791495885409) q[6];
ry(-1.2634512832189309) q[7];
cx q[6],q[7];
ry(1.5714319147951077) q[6];
ry(-0.7778968451271754) q[7];
cx q[6],q[7];
ry(-3.1402726030628685) q[7];
ry(0.020276359910343267) q[8];
cx q[7],q[8];
ry(-0.8258084994151567) q[7];
ry(-1.442685162214988) q[8];
cx q[7],q[8];
ry(3.139185668031079) q[8];
ry(-2.3943144725948584) q[9];
cx q[8],q[9];
ry(-1.5715897411461723) q[8];
ry(1.5596492856167812) q[9];
cx q[8],q[9];
ry(1.5693834408089877) q[9];
ry(-0.6986558302427426) q[10];
cx q[9],q[10];
ry(-3.1402576733310754) q[9];
ry(-2.73617227274029) q[10];
cx q[9],q[10];
ry(0.912571379862313) q[10];
ry(-0.850442662500801) q[11];
cx q[10],q[11];
ry(2.7428668904713467) q[10];
ry(2.766506382801394) q[11];
cx q[10],q[11];
ry(0.8657412917831904) q[0];
ry(0.017189407750991318) q[1];
cx q[0],q[1];
ry(-1.5285811582074555) q[0];
ry(-2.3767482406307603) q[1];
cx q[0],q[1];
ry(1.4425074455292077) q[1];
ry(2.3277648152913826) q[2];
cx q[1],q[2];
ry(-2.875577211003646) q[1];
ry(1.4994118805891592) q[2];
cx q[1],q[2];
ry(0.36822783081655475) q[2];
ry(-2.0843183326946884) q[3];
cx q[2],q[3];
ry(0.20829807855849178) q[2];
ry(-1.6369917268330285) q[3];
cx q[2],q[3];
ry(-3.130185415031225) q[3];
ry(-1.6649197397387) q[4];
cx q[3],q[4];
ry(0.771854083005552) q[3];
ry(1.569779313597829) q[4];
cx q[3],q[4];
ry(2.7712482392848727) q[4];
ry(2.88194183294383) q[5];
cx q[4],q[5];
ry(3.092601095496806) q[4];
ry(-0.17349821845381544) q[5];
cx q[4],q[5];
ry(1.0828279636209153) q[5];
ry(1.5654575908886716) q[6];
cx q[5],q[6];
ry(-3.1027350656367365) q[5];
ry(-1.7263702910774725e-05) q[6];
cx q[5],q[6];
ry(2.211332000218561) q[6];
ry(1.5689384508756872) q[7];
cx q[6],q[7];
ry(2.8859678361824086) q[6];
ry(0.0022919724896070903) q[7];
cx q[6],q[7];
ry(-2.5909996745227613) q[7];
ry(1.5703558552497938) q[8];
cx q[7],q[8];
ry(1.1253451120011617) q[7];
ry(0.0010189047736426937) q[8];
cx q[7],q[8];
ry(-0.32536404789350565) q[8];
ry(1.9550838102571166) q[9];
cx q[8],q[9];
ry(3.1372843755635023) q[8];
ry(0.0009458726214370827) q[9];
cx q[8],q[9];
ry(-1.184918307508335) q[9];
ry(0.166783946889088) q[10];
cx q[9],q[10];
ry(-1.5677191218834772) q[9];
ry(1.9421511638724174) q[10];
cx q[9],q[10];
ry(2.008439495291613) q[10];
ry(-0.45709398528642176) q[11];
cx q[10],q[11];
ry(1.5725501649803033) q[10];
ry(-3.1414004318366864) q[11];
cx q[10],q[11];
ry(2.449729444094757) q[0];
ry(0.2271758072262271) q[1];
ry(-1.7150825314715206) q[2];
ry(1.5716333405920704) q[3];
ry(-1.5582528027302232) q[4];
ry(2.058105294958028) q[5];
ry(0.9240490865610278) q[6];
ry(-0.5534625533412731) q[7];
ry(-2.8161277436767302) q[8];
ry(1.5720151233351445) q[9];
ry(-1.1326665517060688) q[10];
ry(1.573060445151297) q[11];