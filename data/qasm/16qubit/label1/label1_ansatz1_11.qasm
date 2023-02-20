OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.6172836203922856) q[0];
rz(2.737586121693091) q[0];
ry(2.7989764093660003) q[1];
rz(-1.3205236296069354) q[1];
ry(-0.6172243552248166) q[2];
rz(0.39974736861686283) q[2];
ry(0.12315469826983172) q[3];
rz(2.117574200439085) q[3];
ry(1.4200800418239918) q[4];
rz(0.13026634054060476) q[4];
ry(0.03447177009600377) q[5];
rz(-0.25196845065063833) q[5];
ry(0.5275872792163313) q[6];
rz(-0.6622092526216049) q[6];
ry(2.347470766754078) q[7];
rz(-2.727925137778883) q[7];
ry(-2.0430114553802516) q[8];
rz(2.5806310065066875) q[8];
ry(2.325912437220955) q[9];
rz(0.34797331522361225) q[9];
ry(-0.6069146502997675) q[10];
rz(-2.0276189783461427) q[10];
ry(-1.632453339276684) q[11];
rz(1.1964535218877175) q[11];
ry(0.2623836714160639) q[12];
rz(1.2845244461186642) q[12];
ry(-0.0927483500605364) q[13];
rz(-2.218312889590079) q[13];
ry(-0.9500947026647646) q[14];
rz(2.2312378769246446) q[14];
ry(-3.126641821799552) q[15];
rz(2.9801406672912165) q[15];
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
ry(2.029122809258234) q[0];
rz(1.4051543301448373) q[0];
ry(-2.311536220799981) q[1];
rz(-0.5143549479884867) q[1];
ry(1.6287904713806587) q[2];
rz(-0.9520020167035635) q[2];
ry(-3.022094658236253) q[3];
rz(-1.5618011945265728) q[3];
ry(-1.6273714389751834) q[4];
rz(0.4994336742201152) q[4];
ry(2.5067119839884744) q[5];
rz(-2.7705761552804677) q[5];
ry(-3.0444080931791264) q[6];
rz(-1.269430138644104) q[6];
ry(-0.0729201109600968) q[7];
rz(0.5046002533830346) q[7];
ry(1.8557981512701813) q[8];
rz(-1.2266075114307888) q[8];
ry(-1.5532373756046614) q[9];
rz(-0.007243227467387552) q[9];
ry(0.020893592481925186) q[10];
rz(2.8222792250597983) q[10];
ry(-2.2811454585327464) q[11];
rz(-0.3529068278984981) q[11];
ry(1.4243127687874253) q[12];
rz(-1.7808445237075432) q[12];
ry(3.1388994218372437) q[13];
rz(1.177898961404079) q[13];
ry(-2.777009818435627) q[14];
rz(-0.5655403322044599) q[14];
ry(-0.014548363546849963) q[15];
rz(2.3465691153153427) q[15];
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
ry(-2.3595968126386673) q[0];
rz(1.1224491955424591) q[0];
ry(-1.3083196202713296) q[1];
rz(2.9455000337469572) q[1];
ry(-0.22583102070021788) q[2];
rz(1.1891283959153969) q[2];
ry(3.1356701521702903) q[3];
rz(-0.9732047836667681) q[3];
ry(0.7760098872328611) q[4];
rz(-0.15629408620242152) q[4];
ry(0.43429353954840805) q[5];
rz(1.6751138030135424) q[5];
ry(-2.8204101319944295) q[6];
rz(-0.9541155208575185) q[6];
ry(1.0942521588349985) q[7];
rz(1.0343529642406781) q[7];
ry(-0.017319150624935033) q[8];
rz(1.3592295493065172) q[8];
ry(2.1814370876168563) q[9];
rz(1.8782468870345015) q[9];
ry(-1.5504610150811062) q[10];
rz(1.9424751255116224) q[10];
ry(0.5202321427026472) q[11];
rz(-2.2376274213349374) q[11];
ry(-2.947461758067455) q[12];
rz(0.5118943944681833) q[12];
ry(1.9627534587292423) q[13];
rz(-0.0707234807518109) q[13];
ry(-1.3111932548792309) q[14];
rz(2.451759279976729) q[14];
ry(0.08519490053003675) q[15];
rz(-2.6791012159000873) q[15];
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
ry(0.14187093692268868) q[0];
rz(0.1875672968117792) q[0];
ry(-0.9155769477548387) q[1];
rz(0.5767294336656423) q[1];
ry(-0.5804880545295887) q[2];
rz(2.7933827678994922) q[2];
ry(1.986663270606969) q[3];
rz(-1.6318790411530228) q[3];
ry(-2.5693586764856873) q[4];
rz(3.0585084574647317) q[4];
ry(0.08326013558584683) q[5];
rz(-0.7799137422628908) q[5];
ry(-1.0709412045823452) q[6];
rz(-1.9810487035280417) q[6];
ry(-0.39544717169888616) q[7];
rz(-1.0227972472176243) q[7];
ry(0.8436926100271895) q[8];
rz(-0.8253390558726946) q[8];
ry(-1.4074623428463984) q[9];
rz(-0.01239623269557565) q[9];
ry(2.9405891742199977) q[10];
rz(-0.6516350378057671) q[10];
ry(-0.9271988346919913) q[11];
rz(-1.1006575247524575) q[11];
ry(-0.10003379831054371) q[12];
rz(-2.933199091514842) q[12];
ry(-0.02026917320811774) q[13];
rz(0.45806916440609496) q[13];
ry(-1.4129200500088226) q[14];
rz(1.1365961167535772) q[14];
ry(-3.129523525627362) q[15];
rz(2.068921566712668) q[15];
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
ry(-0.6228711563500656) q[0];
rz(1.2421771975665887) q[0];
ry(1.439828572247559) q[1];
rz(-2.308893862166235) q[1];
ry(2.9551459942701457) q[2];
rz(0.5206838970063803) q[2];
ry(0.9559777830854185) q[3];
rz(2.985600714074176) q[3];
ry(-3.092380007390443) q[4];
rz(-0.3174794146226123) q[4];
ry(1.7281959377403509) q[5];
rz(1.8080148045752533) q[5];
ry(2.654416025902931) q[6];
rz(0.738964511278999) q[6];
ry(3.1220054660984577) q[7];
rz(-2.1913437874067663) q[7];
ry(1.5982458577336631) q[8];
rz(0.8090782040062763) q[8];
ry(-0.1989437236048932) q[9];
rz(-1.6221964223936942) q[9];
ry(1.956356790455101) q[10];
rz(1.7951802485305057) q[10];
ry(0.6397219265750413) q[11];
rz(2.5342630265834165) q[11];
ry(-1.6310114881192361) q[12];
rz(-0.6466842172596827) q[12];
ry(0.046774237930989136) q[13];
rz(-1.9616645274610898) q[13];
ry(1.7467663954983157) q[14];
rz(0.30868340222382024) q[14];
ry(2.2803598264751566) q[15];
rz(2.4603816509705383) q[15];
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
ry(-1.3570121845459928) q[0];
rz(-2.0156063120193366) q[0];
ry(-2.13702693996676) q[1];
rz(2.1155808626484207) q[1];
ry(0.18984225461288418) q[2];
rz(-1.6385176349972672) q[2];
ry(2.0278821355033227) q[3];
rz(-0.4025951654649636) q[3];
ry(-0.7321072578428183) q[4];
rz(-2.871183185311386) q[4];
ry(-2.8636246889235415) q[5];
rz(-1.5217418786215227) q[5];
ry(-0.3584945049942747) q[6];
rz(3.0343391016640004) q[6];
ry(-1.5752101410549173) q[7];
rz(-3.130743841349481) q[7];
ry(-3.1342047860652134) q[8];
rz(0.9022403892127002) q[8];
ry(-0.016472388954444958) q[9];
rz(2.537154415270676) q[9];
ry(-2.2410472904803242) q[10];
rz(0.025692110654853908) q[10];
ry(-1.662298989130658) q[11];
rz(-1.7719234188646598) q[11];
ry(3.1247862059279177) q[12];
rz(-0.7509572110792463) q[12];
ry(1.2703434671321467) q[13];
rz(-1.5487476319195133) q[13];
ry(-0.794120921723616) q[14];
rz(-0.9823742632539432) q[14];
ry(2.485498001494443) q[15];
rz(-0.23540021824125645) q[15];
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
ry(-2.0048377892554448) q[0];
rz(-1.9802087149663476) q[0];
ry(0.9011294554031634) q[1];
rz(1.0595175879438248) q[1];
ry(2.234554712408108) q[2];
rz(0.6154941435310347) q[2];
ry(2.6889725472093797) q[3];
rz(2.649771262407796) q[3];
ry(-0.07446823821823743) q[4];
rz(1.6928538773988047) q[4];
ry(-3.068462114771053) q[5];
rz(2.47142364970448) q[5];
ry(-1.58595438605738) q[6];
rz(0.056786462899341535) q[6];
ry(2.176140094581082) q[7];
rz(0.4371762028836309) q[7];
ry(1.1202396363182743) q[8];
rz(-0.04751759605965186) q[8];
ry(0.004703461228914962) q[9];
rz(2.9818224087140015) q[9];
ry(-0.4743924277725425) q[10];
rz(-3.0028641844212633) q[10];
ry(-2.4433960757212114) q[11];
rz(-2.0031975914973974) q[11];
ry(-3.1216693122873673) q[12];
rz(2.269995608509239) q[12];
ry(-0.0723972150744281) q[13];
rz(-0.2037981887819402) q[13];
ry(-1.0883025234762265) q[14];
rz(-2.6831705391081173) q[14];
ry(2.256903143001722) q[15];
rz(0.3697594731166454) q[15];
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
ry(-1.8114421930769495) q[0];
rz(2.053053230584064) q[0];
ry(-3.0534776589184474) q[1];
rz(-1.9589444637000977) q[1];
ry(3.1156622218943104) q[2];
rz(-1.4466667826720674) q[2];
ry(1.8937286675377818) q[3];
rz(2.5090623295116536) q[3];
ry(0.3947939583316664) q[4];
rz(1.9164014627976291) q[4];
ry(1.5590123207150293) q[5];
rz(3.122169027787504) q[5];
ry(0.38323512402140913) q[6];
rz(3.0528231360468374) q[6];
ry(-3.0235923635103465) q[7];
rz(0.4683887171398228) q[7];
ry(2.986217940184319) q[8];
rz(-0.1528262424637069) q[8];
ry(0.0739255260113243) q[9];
rz(-0.2379619220014242) q[9];
ry(2.030262412640929) q[10];
rz(0.13654113123718165) q[10];
ry(1.2568455106386818) q[11];
rz(-2.9844933894349635) q[11];
ry(3.017796745535214) q[12];
rz(0.4461179882334884) q[12];
ry(2.709029547316272) q[13];
rz(2.6925258365175337) q[13];
ry(1.4027314014978334) q[14];
rz(2.3444739175163485) q[14];
ry(3.072462514751006) q[15];
rz(-2.52191986607472) q[15];
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
ry(-0.6167229469011302) q[0];
rz(-1.4393383580422516) q[0];
ry(-1.8654790010756306) q[1];
rz(-2.107132747022404) q[1];
ry(1.9829179751797694) q[2];
rz(-2.659462767644786) q[2];
ry(-1.0819730566244288) q[3];
rz(2.962143796239716) q[3];
ry(1.5735055157222886) q[4];
rz(0.0013056544920058062) q[4];
ry(2.580608485436138) q[5];
rz(-0.0038564392377669066) q[5];
ry(-0.3745053032101078) q[6];
rz(0.736252296937043) q[6];
ry(-2.679818406205318) q[7];
rz(-3.1224261819487165) q[7];
ry(-2.4080859063666193) q[8];
rz(2.9420119798140782) q[8];
ry(1.133708014069849) q[9];
rz(-2.0521452784675973) q[9];
ry(1.3996689887938905) q[10];
rz(-2.175108877962951) q[10];
ry(-2.482129214566206) q[11];
rz(2.491995479806887) q[11];
ry(-2.281486994758378) q[12];
rz(1.1737647969806038) q[12];
ry(-1.6552725386008058) q[13];
rz(-2.7975853218424955) q[13];
ry(0.21618571028412337) q[14];
rz(1.0526998827231444) q[14];
ry(-2.2338066496061106) q[15];
rz(1.5857577388045163) q[15];
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
ry(1.743924965211857) q[0];
rz(2.271034978323182) q[0];
ry(-0.32116971592620513) q[1];
rz(2.8033894595108246) q[1];
ry(-2.6282400692214294) q[2];
rz(-2.5201767862028794) q[2];
ry(-1.5689462932907743) q[3];
rz(-3.1407151248938874) q[3];
ry(-0.6989532548350716) q[4];
rz(-1.0173978969905577) q[4];
ry(-0.002057397221775921) q[5];
rz(1.5917467162985774) q[5];
ry(3.131314165923967) q[6];
rz(0.3458446235861387) q[6];
ry(2.394219269299437) q[7];
rz(-1.0405398004083493) q[7];
ry(-2.9263600175561555) q[8];
rz(-2.6385515656460394) q[8];
ry(-0.05317199110378584) q[9];
rz(2.261937943908019) q[9];
ry(-3.0424444526385552) q[10];
rz(3.1412236344606526) q[10];
ry(3.0370929853888495) q[11];
rz(1.0877307311517084) q[11];
ry(-0.0661895937757065) q[12];
rz(-0.2934867750612681) q[12];
ry(0.8673071110302458) q[13];
rz(-1.5082851257578422) q[13];
ry(3.0370371494006783) q[14];
rz(-0.6113229514925544) q[14];
ry(2.96154274731887) q[15];
rz(-0.3923244720205271) q[15];
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
ry(0.043131937154379685) q[0];
rz(-0.5991399182066087) q[0];
ry(-1.1013249010268273) q[1];
rz(-3.127461712068898) q[1];
ry(1.8929969206824664) q[2];
rz(0.001113680761344129) q[2];
ry(-0.778897618239516) q[3];
rz(-0.6968252381346803) q[3];
ry(1.7097195806229526) q[4];
rz(1.3230631185456598) q[4];
ry(1.5783079561073223) q[5];
rz(-0.6837821730505937) q[5];
ry(2.565355163587942) q[6];
rz(1.1153217684980092) q[6];
ry(-2.943700742515461) q[7];
rz(1.2729622062963195) q[7];
ry(-2.289845891015137) q[8];
rz(-2.186224533043656) q[8];
ry(-2.792086192326996) q[9];
rz(-0.38271169224649615) q[9];
ry(2.2567803176501666) q[10];
rz(-0.9378855483434573) q[10];
ry(-2.697232750943804) q[11];
rz(0.8798461268826456) q[11];
ry(-2.7040749414925207) q[12];
rz(-0.0900591740261838) q[12];
ry(-0.8258100005622246) q[13];
rz(-1.931198387498343) q[13];
ry(-1.8688222259405247) q[14];
rz(1.516111475969163) q[14];
ry(-1.1805994844164927) q[15];
rz(-0.8863027377651513) q[15];
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
ry(1.7297876064430282) q[0];
rz(-0.5906750754160841) q[0];
ry(0.0017917539888411094) q[1];
rz(3.131154942833195) q[1];
ry(0.6789324346771825) q[2];
rz(2.863909463406721) q[2];
ry(-3.1386516676123644) q[3];
rz(-2.2597902294020416) q[3];
ry(-2.9318469338100566) q[4];
rz(-0.035154440326969925) q[4];
ry(-3.140448159363328) q[5];
rz(1.3082946025496476) q[5];
ry(3.0751327197956373) q[6];
rz(-1.885989652802009) q[6];
ry(3.064847431247195) q[7];
rz(1.1697816474221376) q[7];
ry(-3.1404248170398286) q[8];
rz(-0.9932586513678503) q[8];
ry(-0.04380682069620523) q[9];
rz(1.591678956359279) q[9];
ry(0.11964083959306493) q[10];
rz(0.9340556430265039) q[10];
ry(0.2161597736949112) q[11];
rz(-1.4512522253138027) q[11];
ry(3.0436976246620104) q[12];
rz(-1.6432971968588914) q[12];
ry(0.7827056071633086) q[13];
rz(-2.080875043362085) q[13];
ry(-0.2740296308514035) q[14];
rz(-1.8547659197921962) q[14];
ry(-0.8339817850536368) q[15];
rz(2.126856031478262) q[15];
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
ry(-1.6233006368662706) q[0];
rz(-1.7823353000786657) q[0];
ry(2.2096598196846404) q[1];
rz(2.9869553944209786) q[1];
ry(-0.330654154348246) q[2];
rz(1.8350396037463117) q[2];
ry(-1.5744472524862987) q[3];
rz(-1.8111616596812867) q[3];
ry(-0.5688295256073683) q[4];
rz(3.0869889728994875) q[4];
ry(-2.991488353997021) q[5];
rz(-1.7348769788724823) q[5];
ry(-1.0005306712852973) q[6];
rz(2.7474192374733133) q[6];
ry(-1.7832947184534944) q[7];
rz(-1.6966218111005285) q[7];
ry(-1.2311912052661826) q[8];
rz(1.4308161202130671) q[8];
ry(2.283365543518536) q[9];
rz(-0.9219269676199778) q[9];
ry(-0.2738998184201162) q[10];
rz(-1.7911837896492153) q[10];
ry(-0.47901761384448255) q[11];
rz(-1.7651930896388874) q[11];
ry(2.670227377836035) q[12];
rz(2.7869319310003347) q[12];
ry(-1.0268477931757833) q[13];
rz(2.7592304013167266) q[13];
ry(1.944770131877375) q[14];
rz(-2.7587615059190487) q[14];
ry(-2.1566962918367727) q[15];
rz(0.7921007417142188) q[15];
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
ry(1.5700968638100519) q[0];
rz(-1.28124235026363) q[0];
ry(-1.5698374869124887) q[1];
rz(-1.5708785237346774) q[1];
ry(0.20528170637660956) q[2];
rz(-5.120718278295299e-05) q[2];
ry(2.876962512518896) q[3];
rz(3.1053937078318574) q[3];
ry(-3.1409397673943187) q[4];
rz(-1.864433645757984) q[4];
ry(-0.02325343831886214) q[5];
rz(1.7589065765398146) q[5];
ry(-3.117385193692867) q[6];
rz(-2.197238079673615) q[6];
ry(3.0099201176016526) q[7];
rz(3.038930868752636) q[7];
ry(-3.1415799154406896) q[8];
rz(-1.1642032305475851) q[8];
ry(3.0300855855069737) q[9];
rz(3.0918868501522287) q[9];
ry(3.0171558635887874) q[10];
rz(-2.1750458839247817) q[10];
ry(-0.18659934738483208) q[11];
rz(-1.485985183540719) q[11];
ry(0.025541600398077513) q[12];
rz(-1.459734773824165) q[12];
ry(-0.12397592463157993) q[13];
rz(-2.205481401787108) q[13];
ry(-0.48293806593579197) q[14];
rz(-1.876308121064507) q[14];
ry(2.521150774530059) q[15];
rz(1.969113445055979) q[15];
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
ry(9.036479758250948e-06) q[0];
rz(-1.7020378921177333) q[0];
ry(1.5692828862922283) q[1];
rz(1.2940000433276513) q[1];
ry(-1.5664033807057587) q[2];
rz(-1.564076070270402) q[2];
ry(-0.008222494196446384) q[3];
rz(0.9265883931830086) q[3];
ry(-0.007830992885574236) q[4];
rz(1.1267151320230635) q[4];
ry(3.0021598506315095) q[5];
rz(0.5328489144838856) q[5];
ry(-1.6093445074343615) q[6];
rz(2.6077985318367705) q[6];
ry(-2.7579540057219303) q[7];
rz(2.6387176314871668) q[7];
ry(-0.08984889721869749) q[8];
rz(-1.546899861129305) q[8];
ry(0.891265896273356) q[9];
rz(-0.6671419631063991) q[9];
ry(-1.6501403300390995) q[10];
rz(-0.7826072370285838) q[10];
ry(1.3754579988069884) q[11];
rz(1.735214941272558) q[11];
ry(-1.288651791075913) q[12];
rz(1.804498305648507) q[12];
ry(1.0468050213655342) q[13];
rz(-0.22474996420876003) q[13];
ry(1.5923454681668132) q[14];
rz(-0.14105402348435325) q[14];
ry(2.991121501963584) q[15];
rz(-1.6082532312534557) q[15];