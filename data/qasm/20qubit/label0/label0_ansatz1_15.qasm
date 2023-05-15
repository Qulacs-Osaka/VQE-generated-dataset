OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-3.126914579340938) q[0];
rz(-0.8944496342636911) q[0];
ry(-0.8309537298486163) q[1];
rz(-0.11082902861321377) q[1];
ry(-3.1346621496153713) q[2];
rz(1.8347149856624716) q[2];
ry(-1.0256732632115684) q[3];
rz(1.15233707385575) q[3];
ry(-0.0020025193733052425) q[4];
rz(0.450868713851671) q[4];
ry(-1.6957348248588637) q[5];
rz(-1.0995107341137151) q[5];
ry(0.00017430351267400113) q[6];
rz(-0.6593560360971936) q[6];
ry(-1.9562865885741594) q[7];
rz(-0.6702035189236826) q[7];
ry(2.8099647257409814) q[8];
rz(2.684784349545293) q[8];
ry(1.7875195612277623) q[9];
rz(-1.3837455787250676) q[9];
ry(-3.0196803310214317) q[10];
rz(2.4739921549645674) q[10];
ry(0.14485534806820866) q[11];
rz(1.8476536521680105) q[11];
ry(2.8926602732975337) q[12];
rz(3.1330380335293246) q[12];
ry(-0.0073894507900485245) q[13];
rz(-2.572286268514505) q[13];
ry(-0.241260761907081) q[14];
rz(-0.5865825935014201) q[14];
ry(3.14083740665341) q[15];
rz(-0.788793893096287) q[15];
ry(2.7251470350124927) q[16];
rz(0.007059106067202657) q[16];
ry(-0.002991078164189531) q[17];
rz(-2.3114579453801682) q[17];
ry(-0.2224097485016658) q[18];
rz(0.08548961721115414) q[18];
ry(-2.6112685644743383) q[19];
rz(-1.9359592212833538) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.1380389965729383) q[0];
rz(0.21776571239330725) q[0];
ry(-3.062650907808445) q[1];
rz(-1.3925992855520166) q[1];
ry(-0.0033221293969574357) q[2];
rz(1.8783404733130649) q[2];
ry(-0.6303119744524667) q[3];
rz(2.6140520343066322) q[3];
ry(-3.136045832023258) q[4];
rz(1.215948875033269) q[4];
ry(0.19236026326610833) q[5];
rz(-1.6527270914744356) q[5];
ry(-0.014312712298547093) q[6];
rz(-2.565537133791472) q[6];
ry(-0.7685281458956438) q[7];
rz(-2.6327983682994263) q[7];
ry(-0.45766502401782644) q[8];
rz(1.9289526831306518) q[8];
ry(-2.70279705588346) q[9];
rz(-0.8705731897191716) q[9];
ry(2.6132706666737104) q[10];
rz(2.6880527937031804) q[10];
ry(1.2966517056327613) q[11];
rz(0.007849534250823353) q[11];
ry(-0.9033513292391446) q[12];
rz(-1.876739858952332) q[12];
ry(-0.11710718840021084) q[13];
rz(-0.7163129330611779) q[13];
ry(0.41502161817526506) q[14];
rz(-0.7814762289417706) q[14];
ry(2.981295639340767) q[15];
rz(2.776371842919403) q[15];
ry(2.7281520581849894) q[16];
rz(-2.3155074932562125) q[16];
ry(0.4492462452854618) q[17];
rz(1.1350239839466794) q[17];
ry(2.793029706967516) q[18];
rz(-1.6366973998955563) q[18];
ry(-0.6761496919766201) q[19];
rz(-2.2813308191120623) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.081197143850952) q[0];
rz(0.7325693653989792) q[0];
ry(1.9355866563073905) q[1];
rz(1.055046053106607) q[1];
ry(-7.09361791675061e-05) q[2];
rz(0.6286803000034102) q[2];
ry(-0.9229144730140382) q[3];
rz(-2.7203141404962907) q[3];
ry(-2.2384639295049533) q[4];
rz(-0.8189373760641114) q[4];
ry(-1.3633076668861086) q[5];
rz(-2.1511477133442134) q[5];
ry(-3.120956289932826) q[6];
rz(-1.2166249558183884) q[6];
ry(1.8706701826983263) q[7];
rz(-2.0460400165420705) q[7];
ry(0.7693943598525994) q[8];
rz(-1.1200478079934175) q[8];
ry(-3.0830573955354503) q[9];
rz(0.1746825964390026) q[9];
ry(3.13155010591602) q[10];
rz(-0.4406158117822684) q[10];
ry(1.1237635437711582) q[11];
rz(0.007490222664874834) q[11];
ry(0.028012794835448492) q[12];
rz(-3.1357993738251255) q[12];
ry(0.04450366196818046) q[13];
rz(-3.008019474164534) q[13];
ry(-0.23863859318447567) q[14];
rz(0.8276915943641328) q[14];
ry(0.185312770494226) q[15];
rz(0.7831155841957872) q[15];
ry(-3.138655367490511) q[16];
rz(2.0129741210518333) q[16];
ry(-1.903533068589708) q[17];
rz(2.9843114688899632) q[17];
ry(1.5077608325953422) q[18];
rz(2.845807886878206) q[18];
ry(0.058571699868928746) q[19];
rz(1.6005727911455674) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.193424513464185) q[0];
rz(1.5226316493697127) q[0];
ry(1.0782215439226448) q[1];
rz(-1.5857053889664208) q[1];
ry(-0.3589743351598143) q[2];
rz(0.11742771565513975) q[2];
ry(3.12182938400778) q[3];
rz(0.0981505369361182) q[3];
ry(3.126271285331807) q[4];
rz(2.5117040846702463) q[4];
ry(3.055712968809685) q[5];
rz(1.3531164216813814) q[5];
ry(2.9549504579247787) q[6];
rz(-1.2861482925313616) q[6];
ry(2.789886795316483) q[7];
rz(-2.4516181814577815) q[7];
ry(2.9012778548995457) q[8];
rz(1.4650152471639843) q[8];
ry(-2.3676070055657807) q[9];
rz(0.8990546939677568) q[9];
ry(0.4946444392917471) q[10];
rz(0.4742516371412062) q[10];
ry(1.182568712315417) q[11];
rz(2.3619881089768553) q[11];
ry(-0.7492923687012593) q[12];
rz(1.8742814194444346) q[12];
ry(2.976525509688294) q[13];
rz(1.9725492734967593) q[13];
ry(-0.21450791067386712) q[14];
rz(-1.7383905387029321) q[14];
ry(-1.9339302069957127) q[15];
rz(-2.808223577536957) q[15];
ry(0.00027842903279218234) q[16];
rz(-2.115128134871703) q[16];
ry(-0.23170003047116872) q[17];
rz(-2.4848003193965424) q[17];
ry(-2.6549336965488606) q[18];
rz(2.9647348202489856) q[18];
ry(-0.6973712917349516) q[19];
rz(1.4238902695702524) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.430249446023493) q[0];
rz(-0.947194245437581) q[0];
ry(-3.126992371923596) q[1];
rz(-0.42495344039684113) q[1];
ry(3.0765625609239846) q[2];
rz(-1.2340425398098738) q[2];
ry(-1.4186051386272247) q[3];
rz(0.4167301048062102) q[3];
ry(-2.298534121636597) q[4];
rz(1.8099501331760992) q[4];
ry(3.116050261849923) q[5];
rz(2.441017072521494) q[5];
ry(0.7101454251371261) q[6];
rz(2.5471862739223865) q[6];
ry(-2.8766464368581675) q[7];
rz(-2.53931820827666) q[7];
ry(0.8003175843442487) q[8];
rz(-0.28031196434428024) q[8];
ry(0.005781736500067857) q[9];
rz(-1.0686434288621562) q[9];
ry(3.0801916071372095) q[10];
rz(0.6297815807039785) q[10];
ry(0.20189768152236692) q[11];
rz(-0.9737315609178526) q[11];
ry(0.8165018373475895) q[12];
rz(-2.015120844793998) q[12];
ry(0.6251431410093168) q[13];
rz(2.8842404598501883) q[13];
ry(-3.0876479312114924) q[14];
rz(0.39183386555150274) q[14];
ry(-2.879173463361393) q[15];
rz(0.7533808668297004) q[15];
ry(2.2068668473106445) q[16];
rz(0.6409323733766863) q[16];
ry(-3.0932109003047192) q[17];
rz(-0.16620493226779764) q[17];
ry(-0.19034061532833582) q[18];
rz(2.5332921998140825) q[18];
ry(-1.4477399590412092) q[19];
rz(1.2916450749326134) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.7170918059426832) q[0];
rz(-1.7108296183589717) q[0];
ry(-2.525374361478093) q[1];
rz(-2.4482342827896315) q[1];
ry(-0.31289414461847315) q[2];
rz(-0.042394345573060654) q[2];
ry(2.9917773225900204) q[3];
rz(-0.8085443104660461) q[3];
ry(0.00861508306430192) q[4];
rz(2.618740428385028) q[4];
ry(2.4496410244951594) q[5];
rz(-0.4699887605197977) q[5];
ry(-3.105359928825003) q[6];
rz(-1.175916250303698) q[6];
ry(0.18491949392948115) q[7];
rz(0.7489091438343869) q[7];
ry(-0.4834504421580221) q[8];
rz(2.667421411943419) q[8];
ry(1.4209357200625066) q[9];
rz(-2.3424461099378053) q[9];
ry(-1.0400519094318916) q[10];
rz(-1.8485532337019048) q[10];
ry(-2.5139688481286435) q[11];
rz(0.6767430274342258) q[11];
ry(-2.792610157074894) q[12];
rz(-0.8044704348248101) q[12];
ry(-0.7790607946429278) q[13];
rz(0.07873551133835388) q[13];
ry(2.6612415961237406) q[14];
rz(-0.5442637112187836) q[14];
ry(0.0007485090344864987) q[15];
rz(2.127903388878071) q[15];
ry(3.139583669638733) q[16];
rz(-2.668508354176246) q[16];
ry(0.7080588646041146) q[17];
rz(1.1528488495228693) q[17];
ry(-2.6950528218355285) q[18];
rz(-1.3431749525786305) q[18];
ry(-0.3904253780608631) q[19];
rz(2.001351473967815) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.000281735155948) q[0];
rz(1.6743913449788195) q[0];
ry(1.2010096236370678) q[1];
rz(2.8500213896506277) q[1];
ry(-2.8637806429723573) q[2];
rz(-2.4199126760424146) q[2];
ry(-2.95414725375952) q[3];
rz(-0.8844415345932612) q[3];
ry(0.27443858599750415) q[4];
rz(0.6175721971754289) q[4];
ry(-3.0155505507194955) q[5];
rz(-3.1297314587944567) q[5];
ry(-1.9823823269867298) q[6];
rz(3.013577977112649) q[6];
ry(0.4253727384459145) q[7];
rz(-2.8107983767746187) q[7];
ry(1.1760600576705507) q[8];
rz(0.4871831105251886) q[8];
ry(-3.1106943167936) q[9];
rz(3.1227261212476196) q[9];
ry(0.5437993533505834) q[10];
rz(1.5780312723541012) q[10];
ry(1.7646658572249472) q[11];
rz(-0.7174273804208597) q[11];
ry(2.3326438258741233) q[12];
rz(0.29892091227747747) q[12];
ry(-1.4427337976057384) q[13];
rz(-0.3365897395173976) q[13];
ry(-2.6418060895876803) q[14];
rz(-0.19642367250303086) q[14];
ry(2.9611454017133574) q[15];
rz(-2.5304501213617216) q[15];
ry(0.33682322314154245) q[16];
rz(-0.9673892006073332) q[16];
ry(1.731702246166532) q[17];
rz(-2.9974143731002214) q[17];
ry(-1.780228007372239) q[18];
rz(0.8883340925160237) q[18];
ry(2.7292005336816625) q[19];
rz(-2.2874696852112013) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.7910751215014064) q[0];
rz(-1.5458871015351379) q[0];
ry(3.1387353952211297) q[1];
rz(-0.12160413553344698) q[1];
ry(-0.007906511550856088) q[2];
rz(2.0590809533098993) q[2];
ry(2.4042561162931086) q[3];
rz(0.7240154038391243) q[3];
ry(-3.1383303421635205) q[4];
rz(-1.7427260768241641) q[4];
ry(0.17794675542740193) q[5];
rz(-0.1797095313505368) q[5];
ry(2.38827446253337) q[6];
rz(-1.7879161219225899) q[6];
ry(3.129438322615488) q[7];
rz(2.338496260685191) q[7];
ry(-0.33267349166104554) q[8];
rz(0.3504329135675422) q[8];
ry(-2.642952759619543) q[9];
rz(0.8400273464378321) q[9];
ry(0.21484998161601102) q[10];
rz(2.2595851896885373) q[10];
ry(-0.12887600724566717) q[11];
rz(-2.214950144114662) q[11];
ry(-0.23345715809177658) q[12];
rz(-1.3389051784889983) q[12];
ry(-2.9920238932057313) q[13];
rz(2.3741800459802582) q[13];
ry(-3.0782679535398545) q[14];
rz(-0.37293465656187763) q[14];
ry(-3.1410035365593183) q[15];
rz(-0.8193686079774288) q[15];
ry(-0.0005236592490467373) q[16];
rz(0.6984961797123174) q[16];
ry(-1.2931282569249865) q[17];
rz(-0.6216240253677164) q[17];
ry(-1.275034701813966) q[18];
rz(-0.4521713823881503) q[18];
ry(1.959800304182692) q[19];
rz(0.5172891259024434) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.8268111466186427) q[0];
rz(-1.5732231073310592) q[0];
ry(1.2549423025190518) q[1];
rz(-2.160062237497085) q[1];
ry(2.0188121493768243) q[2];
rz(3.0185264837358674) q[2];
ry(-3.060234321520763) q[3];
rz(0.7948377040932753) q[3];
ry(0.8818945565499038) q[4];
rz(2.649340999275104) q[4];
ry(-1.3450446423023676) q[5];
rz(-1.1505104101844061) q[5];
ry(2.019002851856219) q[6];
rz(0.6530782489331309) q[6];
ry(2.1236288656903417) q[7];
rz(-1.4520825246352056) q[7];
ry(-1.638694532018819) q[8];
rz(0.011811679588849523) q[8];
ry(-0.26398190089978907) q[9];
rz(1.2059075836237025) q[9];
ry(-0.08317236300721781) q[10];
rz(-2.292093193493876) q[10];
ry(1.781278096121724) q[11];
rz(2.704584120317576) q[11];
ry(-0.8372270247602103) q[12];
rz(2.6687943720477234) q[12];
ry(-2.8667970598115886) q[13];
rz(-1.157545442201125) q[13];
ry(-0.371150748356901) q[14];
rz(-2.1813169954823923) q[14];
ry(0.09956680460014768) q[15];
rz(-2.572296853884878) q[15];
ry(0.893219745244105) q[16];
rz(-0.9769122995037323) q[16];
ry(-1.5374224508098155) q[17];
rz(-0.15083445058823927) q[17];
ry(-1.4345107150313259) q[18];
rz(0.5920091314728487) q[18];
ry(1.7353221210601206) q[19];
rz(0.5372327975931652) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.079933057912701) q[0];
rz(-3.134186651037628) q[0];
ry(0.19142815843920263) q[1];
rz(2.6292252634868682) q[1];
ry(0.015972033245847634) q[2];
rz(2.812547283744135) q[2];
ry(0.16993110664772718) q[3];
rz(0.10934084451694216) q[3];
ry(0.19858128521468524) q[4];
rz(-0.03065399219445375) q[4];
ry(0.014695508140351414) q[5];
rz(-0.8209586540495923) q[5];
ry(-3.0407323827307056) q[6];
rz(-0.22861500169680216) q[6];
ry(3.10679886484923) q[7];
rz(1.1165397217156219) q[7];
ry(-1.611974929036627) q[8];
rz(1.3603910027508754) q[8];
ry(2.6467073033128177) q[9];
rz(-2.158958661272428) q[9];
ry(1.0321483105384734) q[10];
rz(0.11071619448497039) q[10];
ry(-0.04594605305929026) q[11];
rz(-0.8817798609219443) q[11];
ry(3.0458116125089507) q[12];
rz(2.1902812120388524) q[12];
ry(-2.827791968355031) q[13];
rz(2.4856096310732605) q[13];
ry(-0.5293356713010358) q[14];
rz(2.459908091882641) q[14];
ry(-1.8143828902700079) q[15];
rz(-3.141328268250269) q[15];
ry(-3.1413481740765987) q[16];
rz(0.6843310329680925) q[16];
ry(-1.6293539227928489) q[17];
rz(0.5305761543837589) q[17];
ry(-2.725823873155465) q[18];
rz(-0.47684085467588355) q[18];
ry(-2.2580289579069497) q[19];
rz(0.8012390221213503) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.2235008542310843) q[0];
rz(-3.1387897829094165) q[0];
ry(-3.111638765443551) q[1];
rz(-2.4077919692570102) q[1];
ry(-1.9253112915780415) q[2];
rz(-0.5657113430105367) q[2];
ry(-1.4832954102653542) q[3];
rz(1.69723465500952) q[3];
ry(-1.5017161572533384) q[4];
rz(-2.0197806889439605) q[4];
ry(0.13983760133785075) q[5];
rz(1.8416448931131424) q[5];
ry(-1.3572055630814335) q[6];
rz(2.7511823574906273) q[6];
ry(-2.5407320792605463) q[7];
rz(-0.0815566478887133) q[7];
ry(-0.15765667651067036) q[8];
rz(-1.3758487662593955) q[8];
ry(-3.135717465577823) q[9];
rz(2.6692330842224057) q[9];
ry(-1.2920457362809659) q[10];
rz(-2.1527139345590984) q[10];
ry(0.7979206338120428) q[11];
rz(-1.2976252478732384) q[11];
ry(1.7377119604478237) q[12];
rz(2.4152563883575073) q[12];
ry(1.002194882020326) q[13];
rz(-2.780382175893269) q[13];
ry(0.00042351390965085354) q[14];
rz(-1.1671603083169229) q[14];
ry(-2.194490402794571) q[15];
rz(0.0010192119589396142) q[15];
ry(-3.1370771266911794) q[16];
rz(2.120612102047488) q[16];
ry(2.8653761932312793) q[17];
rz(0.5428467511866569) q[17];
ry(-1.7787689525091441) q[18];
rz(-2.5247543596025355) q[18];
ry(-1.3392498146804601) q[19];
rz(-0.10670524394646373) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.2987061483306017) q[0];
rz(-0.5169125730018601) q[0];
ry(3.10129003131468) q[1];
rz(0.45169442933902776) q[1];
ry(-3.1146361700747067) q[2];
rz(2.640103967098062) q[2];
ry(0.8877948068237743) q[3];
rz(-0.4390648849461087) q[3];
ry(0.5109734078585468) q[4];
rz(1.9016695120323233) q[4];
ry(-3.0696597233746896) q[5];
rz(-2.7704838500990427) q[5];
ry(3.1251146212948617) q[6];
rz(-2.646508768744027) q[6];
ry(3.0905110217394713) q[7];
rz(-0.3200012929418791) q[7];
ry(1.1602926952075983) q[8];
rz(0.2557836748036316) q[8];
ry(0.0971401036750983) q[9];
rz(-2.165617865568526) q[9];
ry(-0.17135872289406645) q[10];
rz(2.304471881672337) q[10];
ry(-3.0842035975145143) q[11];
rz(1.9688747637517532) q[11];
ry(-3.1252290914233476) q[12];
rz(0.9999753995324445) q[12];
ry(2.840290165311776) q[13];
rz(2.043356541249783) q[13];
ry(1.9929769667110901) q[14];
rz(-2.5592841667089568) q[14];
ry(-1.318369420821475) q[15];
rz(2.5154528622056156) q[15];
ry(3.1410381381994994) q[16];
rz(-1.9947203284748545) q[16];
ry(1.9099886995907915) q[17];
rz(2.787609793814769) q[17];
ry(-2.0280353817804735) q[18];
rz(-1.0479668393467563) q[18];
ry(1.097343280802237) q[19];
rz(1.9604995981192406) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.680603413828825) q[0];
rz(1.437865909708228) q[0];
ry(3.007110745019877) q[1];
rz(0.6097059919797241) q[1];
ry(3.0925031069354882) q[2];
rz(1.4832549225156466) q[2];
ry(-0.32306952080274454) q[3];
rz(0.1371973863042948) q[3];
ry(-3.123424252904853) q[4];
rz(2.439317379782573) q[4];
ry(-0.09111611765351668) q[5];
rz(-1.1984805161073495) q[5];
ry(-2.8222701479310808) q[6];
rz(-2.083164551801352) q[6];
ry(1.0383734820214263) q[7];
rz(0.4971543214640156) q[7];
ry(1.4995945801849215) q[8];
rz(0.3687743775484673) q[8];
ry(3.1375909133218642) q[9];
rz(2.1410104966346175) q[9];
ry(-0.7095251643483484) q[10];
rz(3.0660022416245694) q[10];
ry(2.3724660694511734) q[11];
rz(2.5076542156541515) q[11];
ry(-0.0929551679091275) q[12];
rz(-0.1740992346462988) q[12];
ry(-2.875653820980029) q[13];
rz(0.5717015318836997) q[13];
ry(1.8720948563109212) q[14];
rz(1.2148351349116064) q[14];
ry(-2.494688982149555) q[15];
rz(0.2600176644763863) q[15];
ry(0.8473472959865047) q[16];
rz(1.1241216975703605) q[16];
ry(0.8377435905900699) q[17];
rz(-2.5377262141025096) q[17];
ry(-2.701073280951181) q[18];
rz(2.6274000958538952) q[18];
ry(-0.3620537536334281) q[19];
rz(2.7355432972137206) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.5867862814268205) q[0];
rz(2.9738728260595404) q[0];
ry(-0.22759479817105444) q[1];
rz(0.029333112551510613) q[1];
ry(3.0370628336367873) q[2];
rz(2.654251297168533) q[2];
ry(-0.8977718922273404) q[3];
rz(1.9501394733566073) q[3];
ry(2.9934371378013815) q[4];
rz(0.7262298508421112) q[4];
ry(-3.056973684998449) q[5];
rz(-1.983567670844286) q[5];
ry(-0.7120212982884794) q[6];
rz(-0.06775319773420475) q[6];
ry(0.329457949279103) q[7];
rz(0.3920147084165947) q[7];
ry(-0.7019786271910355) q[8];
rz(1.032849627621295) q[8];
ry(3.0163174943807696) q[9];
rz(-2.6182428590505915) q[9];
ry(-1.2445014974725321) q[10];
rz(-0.4108122610110003) q[10];
ry(3.100834155679857) q[11];
rz(-0.24669081206189022) q[11];
ry(0.03264895298749545) q[12];
rz(-2.95036347296305) q[12];
ry(-2.5901687450594966) q[13];
rz(0.15716160596076606) q[13];
ry(-0.2446167196674578) q[14];
rz(-1.3162045771539415) q[14];
ry(-0.0012474280439684904) q[15];
rz(1.3907462026608401) q[15];
ry(-2.2969864309822356) q[16];
rz(2.7064961510309784) q[16];
ry(-0.24085850946972176) q[17];
rz(-0.029281834584472526) q[17];
ry(-1.6788545597955522) q[18];
rz(-0.6140025800328304) q[18];
ry(-0.6964218045304652) q[19];
rz(1.153972041059709) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.6302206354281779) q[0];
rz(2.524689893262179) q[0];
ry(2.8891722895212544) q[1];
rz(-0.5661683617570101) q[1];
ry(-1.3210600830253891) q[2];
rz(-0.5976442699066866) q[2];
ry(0.5235713430059379) q[3];
rz(-1.5929280300134998) q[3];
ry(-1.3257168357295015) q[4];
rz(0.6884799986619178) q[4];
ry(0.02702616052940963) q[5];
rz(2.857416401213228) q[5];
ry(-0.4275345967486972) q[6];
rz(3.1292103073933695) q[6];
ry(2.954606094530476) q[7];
rz(-2.9533245346142136) q[7];
ry(-2.9468221495851177) q[8];
rz(-2.7726927184452888) q[8];
ry(-0.27562869785020805) q[9];
rz(2.6718961614284154) q[9];
ry(2.3501279540574274) q[10];
rz(-0.8886471345209673) q[10];
ry(2.6693944438503494) q[11];
rz(3.1255812785039603) q[11];
ry(1.2992067244749754) q[12];
rz(0.44681408751495355) q[12];
ry(2.9679021569392336) q[13];
rz(-3.000255251534218) q[13];
ry(0.3362222809971599) q[14];
rz(-0.04878578797485744) q[14];
ry(-0.0029352751868154496) q[15];
rz(-0.1999029020839658) q[15];
ry(2.8165276492329934) q[16];
rz(-0.4677484721528146) q[16];
ry(2.7001683509691308) q[17];
rz(-0.621861497670036) q[17];
ry(-0.6733249851123028) q[18];
rz(-2.0021989572693286) q[18];
ry(2.685246409333806) q[19];
rz(-2.037870301788817) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.24781275737458583) q[0];
rz(2.0750528035567295) q[0];
ry(2.7647130344324653) q[1];
rz(-0.8220985516268726) q[1];
ry(0.891634538081076) q[2];
rz(2.018782539162548) q[2];
ry(0.010138010433792194) q[3];
rz(2.5165716901782083) q[3];
ry(-0.1276867183689514) q[4];
rz(-1.9444621724789926) q[4];
ry(-3.136412135457333) q[5];
rz(0.10406862378573134) q[5];
ry(0.66922519724368) q[6];
rz(-2.1272700764758037) q[6];
ry(-0.21787360462010547) q[7];
rz(-0.07876999457406961) q[7];
ry(-2.461801214359049) q[8];
rz(2.998750337698103) q[8];
ry(-0.12746941976790005) q[9];
rz(-1.7578129587128863) q[9];
ry(3.039691190405333) q[10];
rz(3.0681247094442172) q[10];
ry(3.13243770957672) q[11];
rz(2.5335210082973836) q[11];
ry(3.138682322458834) q[12];
rz(0.47494568231244033) q[12];
ry(-0.37119771931016476) q[13];
rz(-1.0597253108345928) q[13];
ry(-2.154536199116753) q[14];
rz(2.780312881862988) q[14];
ry(3.117502281514307) q[15];
rz(-0.05726146805293613) q[15];
ry(0.42266362709184335) q[16];
rz(-2.965040828124852) q[16];
ry(0.6704642934207872) q[17];
rz(0.8061609357970881) q[17];
ry(1.5124076609943522) q[18];
rz(1.2193917013264368) q[18];
ry(2.8405269950963277) q[19];
rz(-2.445221677586209) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.0728760409427656) q[0];
rz(-1.6583852077677972) q[0];
ry(2.700121428213232) q[1];
rz(-0.7684705927278959) q[1];
ry(3.077218816674254) q[2];
rz(3.026958502350504) q[2];
ry(2.994745739535147) q[3];
rz(-0.9893068604157006) q[3];
ry(-0.25336355353619117) q[4];
rz(-1.1825535249881893) q[4];
ry(-0.9390055467631564) q[5];
rz(-1.088510803671418) q[5];
ry(2.1548709542701854) q[6];
rz(1.900574022168464) q[6];
ry(-2.569928735676359) q[7];
rz(-0.9105955305346775) q[7];
ry(-0.5855364590409264) q[8];
rz(-0.0355701849924946) q[8];
ry(0.036601863627350184) q[9];
rz(1.8203142505836596) q[9];
ry(-0.9183863999374343) q[10];
rz(-0.3270192546040427) q[10];
ry(-2.9733128913788165) q[11];
rz(-2.3649448706855667) q[11];
ry(-2.2929377924084737) q[12];
rz(-2.921115881464591) q[12];
ry(-3.107914359895425) q[13];
rz(2.076439018442098) q[13];
ry(3.087646108208098) q[14];
rz(1.4082994335830905) q[14];
ry(-2.3431570528410286) q[15];
rz(-0.006347859259082014) q[15];
ry(1.3836725674852595) q[16];
rz(-0.1449788362462809) q[16];
ry(1.6392860336604314) q[17];
rz(1.683485956835676) q[17];
ry(-0.9868781501904964) q[18];
rz(-0.11916515475665346) q[18];
ry(1.165673683395072) q[19];
rz(-2.447616674368194) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.0961310457414815) q[0];
rz(2.5661393672799884) q[0];
ry(-2.966775090768555) q[1];
rz(2.150292563454432) q[1];
ry(-3.0866144690348842) q[2];
rz(2.151286359556318) q[2];
ry(3.119883153297576) q[3];
rz(-0.852507568857093) q[3];
ry(3.0886865245893715) q[4];
rz(1.7788832542422766) q[4];
ry(0.033963990677322364) q[5];
rz(-2.4397703078942143) q[5];
ry(0.005552195967312379) q[6];
rz(0.3290684617366839) q[6];
ry(0.03831687886599954) q[7];
rz(-3.131769136670927) q[7];
ry(2.470500726418123) q[8];
rz(3.086694047010436) q[8];
ry(0.1327793316994985) q[9];
rz(-2.5070124934973466) q[9];
ry(2.9845526899655606) q[10];
rz(-2.4480965778021253) q[10];
ry(-0.0880774152347472) q[11];
rz(-3.0718680744034) q[11];
ry(-0.05657067610960643) q[12];
rz(-2.124236142149492) q[12];
ry(-0.7086331987674479) q[13];
rz(-2.6714753822745236) q[13];
ry(3.098448883616996) q[14];
rz(0.011852191506914616) q[14];
ry(-3.1189619474239634) q[15];
rz(3.1366706677599594) q[15];
ry(3.0535042350872255) q[16];
rz(-0.06386211300640189) q[16];
ry(-0.003151574965552994) q[17];
rz(2.479888931453535) q[17];
ry(3.118199323301294) q[18];
rz(3.0370959814737697) q[18];
ry(0.28073120610713787) q[19];
rz(-2.140852274919398) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.28702910399910037) q[0];
rz(-1.71441025052799) q[0];
ry(-1.8427724711462314) q[1];
rz(0.1717380933468533) q[1];
ry(-3.071557604184847) q[2];
rz(-0.06394379604031729) q[2];
ry(-2.977422783680773) q[3];
rz(2.387064578606115) q[3];
ry(-2.6715056717348435) q[4];
rz(-2.659105871388609) q[4];
ry(-2.146405919418892) q[5];
rz(1.4469401523199101) q[5];
ry(2.5401479794462554) q[6];
rz(-2.4950890339400633) q[6];
ry(0.06820238949998458) q[7];
rz(-2.0569909432021083) q[7];
ry(2.160710902605077) q[8];
rz(-3.069152821808027) q[8];
ry(1.8828260547204394) q[9];
rz(3.1405759478795816) q[9];
ry(-2.8545989851348175) q[10];
rz(-2.6152805410477624) q[10];
ry(-1.995325955272273) q[11];
rz(2.8147779531710517) q[11];
ry(0.5587554014371572) q[12];
rz(-0.43122495715513626) q[12];
ry(-3.1092049507989725) q[13];
rz(-1.8142872754541328) q[13];
ry(2.8441658728366805) q[14];
rz(-1.641348409347598) q[14];
ry(0.7535579188515449) q[15];
rz(1.569281762782721) q[15];
ry(1.9468578281707307) q[16];
rz(1.4883006309279723) q[16];
ry(0.09140839376735352) q[17];
rz(2.0884670636546856) q[17];
ry(2.1097529987250496) q[18];
rz(-1.5471252865451257) q[18];
ry(1.2680509051516333) q[19];
rz(1.100552313176965) q[19];