OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.2910436362753863) q[0];
ry(1.1832967674756523) q[1];
cx q[0],q[1];
ry(1.5822386501236612) q[0];
ry(-1.6078821203786207) q[1];
cx q[0],q[1];
ry(-2.625516838018627) q[2];
ry(-1.6418191438198786) q[3];
cx q[2],q[3];
ry(1.8352887390007357) q[2];
ry(2.5777154988579385) q[3];
cx q[2],q[3];
ry(1.1299444233984235) q[4];
ry(-3.104542086324974) q[5];
cx q[4],q[5];
ry(2.22173584247437) q[4];
ry(2.9113171018037947) q[5];
cx q[4],q[5];
ry(-0.33441019948622847) q[6];
ry(-2.8283140545294785) q[7];
cx q[6],q[7];
ry(1.322228094181227) q[6];
ry(3.050061468262798) q[7];
cx q[6],q[7];
ry(-1.8889170921602227) q[8];
ry(-2.375825403747348) q[9];
cx q[8],q[9];
ry(2.2732495371894954) q[8];
ry(-2.2866290450432256) q[9];
cx q[8],q[9];
ry(1.5682637135456186) q[10];
ry(0.6543193271793619) q[11];
cx q[10],q[11];
ry(-0.18459774525802394) q[10];
ry(0.3032358172876295) q[11];
cx q[10],q[11];
ry(-3.0785039749946264) q[0];
ry(2.6903629592272544) q[2];
cx q[0],q[2];
ry(0.20150828474830706) q[0];
ry(-2.391622727603869) q[2];
cx q[0],q[2];
ry(-2.3389702209872976) q[2];
ry(-3.048632359084974) q[4];
cx q[2],q[4];
ry(1.0903229542381574) q[2];
ry(-3.0466510730977028) q[4];
cx q[2],q[4];
ry(-0.5259984569977803) q[4];
ry(1.1443760002144276) q[6];
cx q[4],q[6];
ry(-1.815036822526186) q[4];
ry(0.7714804666574856) q[6];
cx q[4],q[6];
ry(1.5689480116508439) q[6];
ry(1.0764698775053665) q[8];
cx q[6],q[8];
ry(1.4809675547135903) q[6];
ry(-2.265807245811029) q[8];
cx q[6],q[8];
ry(0.3565805380743106) q[8];
ry(-3.0541233141701682) q[10];
cx q[8],q[10];
ry(0.3030739726001693) q[8];
ry(0.4112399734123118) q[10];
cx q[8],q[10];
ry(-0.6350468330796342) q[1];
ry(-1.4259749608176648) q[3];
cx q[1],q[3];
ry(3.0470595240805065) q[1];
ry(-1.7578508755546567) q[3];
cx q[1],q[3];
ry(0.18273271124745585) q[3];
ry(1.3505221285498985) q[5];
cx q[3],q[5];
ry(-1.3867827567127848) q[3];
ry(-1.7941346862613679) q[5];
cx q[3],q[5];
ry(2.7314357009392625) q[5];
ry(-1.525642736199918) q[7];
cx q[5],q[7];
ry(-3.0841440919420786) q[5];
ry(0.7374979788157164) q[7];
cx q[5],q[7];
ry(0.04461555309975261) q[7];
ry(1.4470883998849808) q[9];
cx q[7],q[9];
ry(-0.9818285336006425) q[7];
ry(0.28556797516681165) q[9];
cx q[7],q[9];
ry(2.627940898005412) q[9];
ry(2.9483124968925702) q[11];
cx q[9],q[11];
ry(2.821141187799013) q[9];
ry(1.7088268389017423) q[11];
cx q[9],q[11];
ry(0.8615007930682985) q[0];
ry(-0.8770583494205664) q[1];
cx q[0],q[1];
ry(-2.1421239910297407) q[0];
ry(2.6258183468648952) q[1];
cx q[0],q[1];
ry(0.36978355261093365) q[2];
ry(-1.938500198323517) q[3];
cx q[2],q[3];
ry(1.4466972375351732) q[2];
ry(-2.1652075762403755) q[3];
cx q[2],q[3];
ry(0.40160856089197355) q[4];
ry(2.6326698977273764) q[5];
cx q[4],q[5];
ry(-0.9833906891144686) q[4];
ry(2.5552927871965934) q[5];
cx q[4],q[5];
ry(2.278884026477356) q[6];
ry(-1.9594406615974571) q[7];
cx q[6],q[7];
ry(-1.396065313582773) q[6];
ry(0.8070679243592869) q[7];
cx q[6],q[7];
ry(1.0466449556599835) q[8];
ry(-2.7716463250426604) q[9];
cx q[8],q[9];
ry(2.016549219157305) q[8];
ry(-1.189692781061385) q[9];
cx q[8],q[9];
ry(0.9694678742828402) q[10];
ry(-1.056163680529818) q[11];
cx q[10],q[11];
ry(-0.54984410289115) q[10];
ry(-1.0646563284250048) q[11];
cx q[10],q[11];
ry(0.016517795205679118) q[0];
ry(2.9227925068508065) q[2];
cx q[0],q[2];
ry(-1.9963422628079812) q[0];
ry(0.7681667094278986) q[2];
cx q[0],q[2];
ry(-0.39405495855397854) q[2];
ry(-0.961895756387053) q[4];
cx q[2],q[4];
ry(1.8640053418790865) q[2];
ry(1.1693717223716549) q[4];
cx q[2],q[4];
ry(-0.37803113146821143) q[4];
ry(-0.6296502215449697) q[6];
cx q[4],q[6];
ry(0.41019538222314106) q[4];
ry(-1.5956067054730159) q[6];
cx q[4],q[6];
ry(2.0384099914126903) q[6];
ry(-0.6180738159062562) q[8];
cx q[6],q[8];
ry(-2.445517851926114) q[6];
ry(2.8251214931105473) q[8];
cx q[6],q[8];
ry(3.003123775292937) q[8];
ry(-2.5292904344317098) q[10];
cx q[8],q[10];
ry(2.8468363263596714) q[8];
ry(-1.3844244562513301) q[10];
cx q[8],q[10];
ry(-1.497986833923782) q[1];
ry(2.1507334084655056) q[3];
cx q[1],q[3];
ry(-0.4291862304848326) q[1];
ry(-1.8886111363948317) q[3];
cx q[1],q[3];
ry(-2.405456580772407) q[3];
ry(-2.4942729070604877) q[5];
cx q[3],q[5];
ry(-1.9076997342871627) q[3];
ry(1.8576586365610266) q[5];
cx q[3],q[5];
ry(1.8100834397351069) q[5];
ry(1.8641513253665427) q[7];
cx q[5],q[7];
ry(-0.41956311091449194) q[5];
ry(1.807170072969999) q[7];
cx q[5],q[7];
ry(1.360979661644416) q[7];
ry(1.3559567080757182) q[9];
cx q[7],q[9];
ry(-1.7487987036787023) q[7];
ry(-2.0158191383933426) q[9];
cx q[7],q[9];
ry(2.802457214981374) q[9];
ry(0.6600411216326201) q[11];
cx q[9],q[11];
ry(1.545410778997777) q[9];
ry(-0.7087950014806239) q[11];
cx q[9],q[11];
ry(0.7683867680093961) q[0];
ry(-2.4204949735063144) q[1];
cx q[0],q[1];
ry(-2.107049557296396) q[0];
ry(0.80723501428404) q[1];
cx q[0],q[1];
ry(3.0059428346930863) q[2];
ry(2.6708639408833426) q[3];
cx q[2],q[3];
ry(-2.2613357854316245) q[2];
ry(-0.3416161741229011) q[3];
cx q[2],q[3];
ry(-0.11963901944747857) q[4];
ry(2.165178591993188) q[5];
cx q[4],q[5];
ry(0.6086488161411641) q[4];
ry(-1.7039303150339284) q[5];
cx q[4],q[5];
ry(-2.0777106450584144) q[6];
ry(0.23798120642255863) q[7];
cx q[6],q[7];
ry(-0.25128770045504667) q[6];
ry(1.6772152014009645) q[7];
cx q[6],q[7];
ry(-2.231581570394286) q[8];
ry(-0.6018368346837394) q[9];
cx q[8],q[9];
ry(-0.5114243923376367) q[8];
ry(2.2613280681769643) q[9];
cx q[8],q[9];
ry(-1.5650538676355883) q[10];
ry(-1.817142209572343) q[11];
cx q[10],q[11];
ry(-1.5516437846746012) q[10];
ry(1.8582683853965691) q[11];
cx q[10],q[11];
ry(-1.2611432742886812) q[0];
ry(0.2726742503806099) q[2];
cx q[0],q[2];
ry(1.9683147535534018) q[0];
ry(1.9537823423319631) q[2];
cx q[0],q[2];
ry(0.47952386042135386) q[2];
ry(-1.410972233755251) q[4];
cx q[2],q[4];
ry(-2.600382831622465) q[2];
ry(-1.6612060880634607) q[4];
cx q[2],q[4];
ry(2.3899992065983233) q[4];
ry(2.6797157382062586) q[6];
cx q[4],q[6];
ry(-1.2498833614674905) q[4];
ry(-1.941967270660316) q[6];
cx q[4],q[6];
ry(-0.03693556274576615) q[6];
ry(-0.19272993488608478) q[8];
cx q[6],q[8];
ry(-1.9432141676780965) q[6];
ry(-0.14309811269313322) q[8];
cx q[6],q[8];
ry(-2.412317498149351) q[8];
ry(1.4220060136198496) q[10];
cx q[8],q[10];
ry(0.2457757878425273) q[8];
ry(0.9849054061574876) q[10];
cx q[8],q[10];
ry(2.346186495496932) q[1];
ry(-0.7720979251371236) q[3];
cx q[1],q[3];
ry(-0.5313738768679732) q[1];
ry(-2.9111635167465066) q[3];
cx q[1],q[3];
ry(-2.3217256073746646) q[3];
ry(-0.10824525155238884) q[5];
cx q[3],q[5];
ry(0.9376332215932663) q[3];
ry(1.1637587571152923) q[5];
cx q[3],q[5];
ry(-2.963045753363354) q[5];
ry(2.49236790179968) q[7];
cx q[5],q[7];
ry(1.9182420708669694) q[5];
ry(-0.5725735649589985) q[7];
cx q[5],q[7];
ry(2.5340137577137223) q[7];
ry(-0.9968365129293577) q[9];
cx q[7],q[9];
ry(-1.658837043185084) q[7];
ry(-2.862076127280051) q[9];
cx q[7],q[9];
ry(2.8922116769378996) q[9];
ry(2.0306323634139156) q[11];
cx q[9],q[11];
ry(2.611506378649651) q[9];
ry(-0.9861955561628559) q[11];
cx q[9],q[11];
ry(2.433947258063961) q[0];
ry(-0.781492432694864) q[1];
cx q[0],q[1];
ry(1.044720585658884) q[0];
ry(-1.001044514482051) q[1];
cx q[0],q[1];
ry(-0.42355672036602776) q[2];
ry(1.2206433702944555) q[3];
cx q[2],q[3];
ry(1.4228112877567347) q[2];
ry(-1.9842336320736973) q[3];
cx q[2],q[3];
ry(-2.1695669085348217) q[4];
ry(-0.4392193979038286) q[5];
cx q[4],q[5];
ry(-2.494547278142588) q[4];
ry(-1.7589580094660944) q[5];
cx q[4],q[5];
ry(0.6751585358967443) q[6];
ry(0.9693807848732456) q[7];
cx q[6],q[7];
ry(1.4595229490206387) q[6];
ry(-1.410644185896175) q[7];
cx q[6],q[7];
ry(0.8424209933999345) q[8];
ry(2.470734399167475) q[9];
cx q[8],q[9];
ry(1.0403795596943324) q[8];
ry(2.945417428893383) q[9];
cx q[8],q[9];
ry(1.8701089006690725) q[10];
ry(-0.596371883381841) q[11];
cx q[10],q[11];
ry(-2.6087182452506608) q[10];
ry(-2.716533382204355) q[11];
cx q[10],q[11];
ry(-2.1780445715591927) q[0];
ry(1.1383223830155709) q[2];
cx q[0],q[2];
ry(1.138422043418563) q[0];
ry(-0.672676722510217) q[2];
cx q[0],q[2];
ry(-0.6912738253120168) q[2];
ry(3.052332669209666) q[4];
cx q[2],q[4];
ry(-0.08252882757358648) q[2];
ry(0.3927658293696651) q[4];
cx q[2],q[4];
ry(-2.014537461690223) q[4];
ry(1.201783446449492) q[6];
cx q[4],q[6];
ry(-2.772172229793703) q[4];
ry(-1.8944280979550336) q[6];
cx q[4],q[6];
ry(3.032684019464254) q[6];
ry(1.7006544440064308) q[8];
cx q[6],q[8];
ry(2.973029846159345) q[6];
ry(1.7694919764214605) q[8];
cx q[6],q[8];
ry(1.0962568146434843) q[8];
ry(1.5921995776356392) q[10];
cx q[8],q[10];
ry(-2.147288382175515) q[8];
ry(-0.9894300147728572) q[10];
cx q[8],q[10];
ry(1.0147588436190125) q[1];
ry(-2.9868473376949134) q[3];
cx q[1],q[3];
ry(0.6450397607681495) q[1];
ry(1.8842088051199903) q[3];
cx q[1],q[3];
ry(2.191472702353807) q[3];
ry(0.848066190798213) q[5];
cx q[3],q[5];
ry(-1.3624162246679676) q[3];
ry(2.312685232991112) q[5];
cx q[3],q[5];
ry(1.9240568006456156) q[5];
ry(1.5579547432591436) q[7];
cx q[5],q[7];
ry(-2.098436608361621) q[5];
ry(2.313548021040648) q[7];
cx q[5],q[7];
ry(-0.2698038596654002) q[7];
ry(2.6501435210544915) q[9];
cx q[7],q[9];
ry(1.3952445967189144) q[7];
ry(-1.893490940069333) q[9];
cx q[7],q[9];
ry(2.3738116182527667) q[9];
ry(-1.6191922771714788) q[11];
cx q[9],q[11];
ry(-1.8857802491629787) q[9];
ry(-0.5489134363736327) q[11];
cx q[9],q[11];
ry(2.896087701115934) q[0];
ry(-1.0781040400842912) q[1];
cx q[0],q[1];
ry(0.7491841927524385) q[0];
ry(-2.5661367128184662) q[1];
cx q[0],q[1];
ry(1.6682243331127176) q[2];
ry(-1.793812609826931) q[3];
cx q[2],q[3];
ry(-2.7287060596399395) q[2];
ry(1.789197634676814) q[3];
cx q[2],q[3];
ry(-0.9951143761877098) q[4];
ry(1.6791962597012553) q[5];
cx q[4],q[5];
ry(0.45166573958227646) q[4];
ry(-1.1608211180913548) q[5];
cx q[4],q[5];
ry(-1.2938881688315416) q[6];
ry(2.7198941064016844) q[7];
cx q[6],q[7];
ry(0.6885467261328069) q[6];
ry(-1.4833497660160297) q[7];
cx q[6],q[7];
ry(-2.3594771472342724) q[8];
ry(2.3704209686304316) q[9];
cx q[8],q[9];
ry(-2.2699370880462575) q[8];
ry(-2.076606991749277) q[9];
cx q[8],q[9];
ry(-0.877172165593467) q[10];
ry(-0.36119136458535994) q[11];
cx q[10],q[11];
ry(-1.376628249258299) q[10];
ry(-1.6388076845096726) q[11];
cx q[10],q[11];
ry(3.103804546558938) q[0];
ry(-1.1584892363285135) q[2];
cx q[0],q[2];
ry(-1.4909598629506027) q[0];
ry(-1.4568321998671756) q[2];
cx q[0],q[2];
ry(-0.2905097414503014) q[2];
ry(-2.533945803711918) q[4];
cx q[2],q[4];
ry(1.6394476508046711) q[2];
ry(2.142345123582711) q[4];
cx q[2],q[4];
ry(0.5763944731233642) q[4];
ry(-1.6199433731256512) q[6];
cx q[4],q[6];
ry(-0.2989086057207171) q[4];
ry(-3.0374881196838244) q[6];
cx q[4],q[6];
ry(-2.5936999137071104) q[6];
ry(-3.07071671387777) q[8];
cx q[6],q[8];
ry(-2.401955006114446) q[6];
ry(1.9111593430687384) q[8];
cx q[6],q[8];
ry(-1.5857586061318605) q[8];
ry(-0.07445187651249835) q[10];
cx q[8],q[10];
ry(-0.1678219634401033) q[8];
ry(2.70373238185112) q[10];
cx q[8],q[10];
ry(-1.7605606934894835) q[1];
ry(-2.256126813413326) q[3];
cx q[1],q[3];
ry(-2.9501254293377657) q[1];
ry(2.755970299484405) q[3];
cx q[1],q[3];
ry(0.33607105321055025) q[3];
ry(0.07037815456667279) q[5];
cx q[3],q[5];
ry(-2.8064812506913657) q[3];
ry(0.08051817393037426) q[5];
cx q[3],q[5];
ry(2.7157596791333134) q[5];
ry(2.410899171147324) q[7];
cx q[5],q[7];
ry(1.3387872992514167) q[5];
ry(3.00474790010135) q[7];
cx q[5],q[7];
ry(1.6446611241283247) q[7];
ry(2.71475274824119) q[9];
cx q[7],q[9];
ry(0.9765174051967402) q[7];
ry(0.9647006637571724) q[9];
cx q[7],q[9];
ry(1.8760177030313792) q[9];
ry(0.5549279320575814) q[11];
cx q[9],q[11];
ry(-0.5729042961674624) q[9];
ry(-1.6528010620358806) q[11];
cx q[9],q[11];
ry(2.6158542085274905) q[0];
ry(2.267590841762627) q[1];
cx q[0],q[1];
ry(2.860130516298488) q[0];
ry(2.6933724428848933) q[1];
cx q[0],q[1];
ry(-0.4818364863933657) q[2];
ry(1.852480860155616) q[3];
cx q[2],q[3];
ry(-2.17181205708561) q[2];
ry(-1.3694169369958946) q[3];
cx q[2],q[3];
ry(2.73415685209463) q[4];
ry(-2.039626130988818) q[5];
cx q[4],q[5];
ry(-2.411476931021039) q[4];
ry(2.9950155911785297) q[5];
cx q[4],q[5];
ry(-1.1693354148407318) q[6];
ry(-2.5724369814135155) q[7];
cx q[6],q[7];
ry(-2.231692807471105) q[6];
ry(0.19134003040958536) q[7];
cx q[6],q[7];
ry(2.927399091358431) q[8];
ry(-2.748597487777916) q[9];
cx q[8],q[9];
ry(0.7506189365558371) q[8];
ry(-1.7432153464978397) q[9];
cx q[8],q[9];
ry(-0.8542529423237522) q[10];
ry(-0.36014665192934103) q[11];
cx q[10],q[11];
ry(1.6244785592430364) q[10];
ry(-0.31215433520116437) q[11];
cx q[10],q[11];
ry(2.786403901967927) q[0];
ry(1.46718017337151) q[2];
cx q[0],q[2];
ry(-0.6545602754074498) q[0];
ry(2.775748772830628) q[2];
cx q[0],q[2];
ry(-0.04844259750426379) q[2];
ry(-0.1607204885938868) q[4];
cx q[2],q[4];
ry(-1.9792704740989773) q[2];
ry(-1.5713362586381256) q[4];
cx q[2],q[4];
ry(0.37046109930707194) q[4];
ry(-0.20451884190978548) q[6];
cx q[4],q[6];
ry(2.2001748408235082) q[4];
ry(-0.5010701249237983) q[6];
cx q[4],q[6];
ry(-0.11992140341016888) q[6];
ry(0.511136784591532) q[8];
cx q[6],q[8];
ry(0.44799337737735095) q[6];
ry(2.7287496814652488) q[8];
cx q[6],q[8];
ry(-2.8359661488626022) q[8];
ry(0.9437626413221185) q[10];
cx q[8],q[10];
ry(0.20377948415896063) q[8];
ry(-1.1940621968535794) q[10];
cx q[8],q[10];
ry(2.8189574508802457) q[1];
ry(-0.12816866703436425) q[3];
cx q[1],q[3];
ry(2.793306959076367) q[1];
ry(1.6849932193787485) q[3];
cx q[1],q[3];
ry(2.1244539694834983) q[3];
ry(-0.4148364569273717) q[5];
cx q[3],q[5];
ry(-2.719652552619697) q[3];
ry(2.5064107783952423) q[5];
cx q[3],q[5];
ry(2.630140344385352) q[5];
ry(-2.791708830072258) q[7];
cx q[5],q[7];
ry(-0.8340399876180244) q[5];
ry(0.6557933184823199) q[7];
cx q[5],q[7];
ry(-1.0790518927014272) q[7];
ry(-2.080088054598149) q[9];
cx q[7],q[9];
ry(-1.2631673193253112) q[7];
ry(2.543061829726716) q[9];
cx q[7],q[9];
ry(2.0614422886730353) q[9];
ry(-0.8774622127370632) q[11];
cx q[9],q[11];
ry(0.537838361139137) q[9];
ry(-1.6960932295687687) q[11];
cx q[9],q[11];
ry(-0.7192374936899871) q[0];
ry(3.136397247599141) q[1];
cx q[0],q[1];
ry(1.1483285619938295) q[0];
ry(-1.1466041968768348) q[1];
cx q[0],q[1];
ry(-2.2702603866544653) q[2];
ry(-2.9447449888507373) q[3];
cx q[2],q[3];
ry(1.0461365957252777) q[2];
ry(-1.9019322179769413) q[3];
cx q[2],q[3];
ry(-1.6931207009461007) q[4];
ry(-0.68576483030873) q[5];
cx q[4],q[5];
ry(-0.7449884304726968) q[4];
ry(-2.2546547504733505) q[5];
cx q[4],q[5];
ry(3.023165480349959) q[6];
ry(-1.765658345550917) q[7];
cx q[6],q[7];
ry(2.7288131513481373) q[6];
ry(0.46985795051714996) q[7];
cx q[6],q[7];
ry(-0.12121064489432569) q[8];
ry(-0.7180355544436801) q[9];
cx q[8],q[9];
ry(-0.34740411507102154) q[8];
ry(-2.3332818233887824) q[9];
cx q[8],q[9];
ry(-0.08306375835999309) q[10];
ry(0.2684243544321294) q[11];
cx q[10],q[11];
ry(2.9776593193968863) q[10];
ry(0.9416620583531543) q[11];
cx q[10],q[11];
ry(2.6207555071982713) q[0];
ry(2.514766373545558) q[2];
cx q[0],q[2];
ry(2.2782608711752337) q[0];
ry(2.8710170899463425) q[2];
cx q[0],q[2];
ry(-1.319028319480837) q[2];
ry(0.9673018135560834) q[4];
cx q[2],q[4];
ry(1.6694135471389675) q[2];
ry(0.8365073955050084) q[4];
cx q[2],q[4];
ry(-0.2657137555447951) q[4];
ry(0.22279652299690586) q[6];
cx q[4],q[6];
ry(-1.9703090096568556) q[4];
ry(2.6826555518382995) q[6];
cx q[4],q[6];
ry(-0.455813754672392) q[6];
ry(2.6738874806231117) q[8];
cx q[6],q[8];
ry(-1.768100182694588) q[6];
ry(1.861457654312902) q[8];
cx q[6],q[8];
ry(-0.9608445540193425) q[8];
ry(-2.1623852768631444) q[10];
cx q[8],q[10];
ry(-1.5769225318099973) q[8];
ry(1.6945245834851994) q[10];
cx q[8],q[10];
ry(-0.3502420420625842) q[1];
ry(0.44123170273660794) q[3];
cx q[1],q[3];
ry(2.469565868750104) q[1];
ry(-1.3454360687382554) q[3];
cx q[1],q[3];
ry(1.5929175648072855) q[3];
ry(-1.6034913835167064) q[5];
cx q[3],q[5];
ry(-0.6195329515068291) q[3];
ry(-1.8548057334492123) q[5];
cx q[3],q[5];
ry(-2.5223927311726206) q[5];
ry(-1.271354807314615) q[7];
cx q[5],q[7];
ry(2.082245530910479) q[5];
ry(-1.6065056310102817) q[7];
cx q[5],q[7];
ry(2.4917491797882914) q[7];
ry(-1.064663689594628) q[9];
cx q[7],q[9];
ry(-2.3729583192955137) q[7];
ry(-1.9263313633323702) q[9];
cx q[7],q[9];
ry(-1.6310754992883718) q[9];
ry(0.016817044952723847) q[11];
cx q[9],q[11];
ry(-2.939067673852303) q[9];
ry(1.4279248239141547) q[11];
cx q[9],q[11];
ry(0.888518619679851) q[0];
ry(1.1116519180167541) q[1];
cx q[0],q[1];
ry(-0.606511589590912) q[0];
ry(1.1792927773842896) q[1];
cx q[0],q[1];
ry(-1.530229968376589) q[2];
ry(0.8587722566682902) q[3];
cx q[2],q[3];
ry(0.37050313838507193) q[2];
ry(-0.8734428321365186) q[3];
cx q[2],q[3];
ry(0.057523096659950916) q[4];
ry(-1.3067185309835625) q[5];
cx q[4],q[5];
ry(-0.5287922497288338) q[4];
ry(-1.9687588647820895) q[5];
cx q[4],q[5];
ry(3.139683125125464) q[6];
ry(1.0137378136353385) q[7];
cx q[6],q[7];
ry(2.567327449205014) q[6];
ry(2.170611572999352) q[7];
cx q[6],q[7];
ry(-1.0400105309656729) q[8];
ry(-2.144219930683051) q[9];
cx q[8],q[9];
ry(-2.3459372683535333) q[8];
ry(0.449199327342726) q[9];
cx q[8],q[9];
ry(-0.9573535877322588) q[10];
ry(0.6933030269101076) q[11];
cx q[10],q[11];
ry(-1.0664845550889792) q[10];
ry(0.34643419982519946) q[11];
cx q[10],q[11];
ry(-1.351358995439841) q[0];
ry(1.3150070263787572) q[2];
cx q[0],q[2];
ry(-0.3537734820171429) q[0];
ry(-1.4720861109883085) q[2];
cx q[0],q[2];
ry(-2.1687909562363865) q[2];
ry(0.9129965964919782) q[4];
cx q[2],q[4];
ry(0.9071998704073465) q[2];
ry(-0.9176257689529224) q[4];
cx q[2],q[4];
ry(-1.1419800249539862) q[4];
ry(2.734466047985324) q[6];
cx q[4],q[6];
ry(2.021873183576491) q[4];
ry(-3.0023142563424674) q[6];
cx q[4],q[6];
ry(1.2915703079655871) q[6];
ry(2.12789383570938) q[8];
cx q[6],q[8];
ry(1.6266379240631235) q[6];
ry(-0.16330951782922298) q[8];
cx q[6],q[8];
ry(0.3837720888342657) q[8];
ry(2.350711440739043) q[10];
cx q[8],q[10];
ry(2.4233698050483072) q[8];
ry(1.973889981765378) q[10];
cx q[8],q[10];
ry(0.5116485582038397) q[1];
ry(0.7630491329153974) q[3];
cx q[1],q[3];
ry(-2.131632866671368) q[1];
ry(2.6781023060025384) q[3];
cx q[1],q[3];
ry(2.8098571699529877) q[3];
ry(-2.1957824605474956) q[5];
cx q[3],q[5];
ry(0.5474370703067266) q[3];
ry(-2.491543962133755) q[5];
cx q[3],q[5];
ry(2.238821900368697) q[5];
ry(-2.0066823037123376) q[7];
cx q[5],q[7];
ry(0.23715154443273437) q[5];
ry(3.0111949050597544) q[7];
cx q[5],q[7];
ry(0.40202059859111916) q[7];
ry(1.1482688133664682) q[9];
cx q[7],q[9];
ry(0.5410549214808278) q[7];
ry(-2.3138560727127535) q[9];
cx q[7],q[9];
ry(-0.45136915635192043) q[9];
ry(-0.8044874872826423) q[11];
cx q[9],q[11];
ry(2.396046048209757) q[9];
ry(-1.257898715340293) q[11];
cx q[9],q[11];
ry(-2.63395777466593) q[0];
ry(2.250896242202578) q[1];
cx q[0],q[1];
ry(0.5850734335144461) q[0];
ry(0.426587672897969) q[1];
cx q[0],q[1];
ry(-2.6211192939103074) q[2];
ry(3.0187078913644756) q[3];
cx q[2],q[3];
ry(1.1068204332403713) q[2];
ry(-1.3950263589364011) q[3];
cx q[2],q[3];
ry(-2.418835703508568) q[4];
ry(-0.5608819379065624) q[5];
cx q[4],q[5];
ry(-2.7854997798948027) q[4];
ry(1.0425575620409129) q[5];
cx q[4],q[5];
ry(0.00978408743969439) q[6];
ry(0.68617159398101) q[7];
cx q[6],q[7];
ry(1.2808332060653689) q[6];
ry(-0.3567138890750877) q[7];
cx q[6],q[7];
ry(1.343567930358578) q[8];
ry(2.296079782887334) q[9];
cx q[8],q[9];
ry(0.8701865980026602) q[8];
ry(-2.5617669706967616) q[9];
cx q[8],q[9];
ry(-1.7511301801213826) q[10];
ry(2.437041380800957) q[11];
cx q[10],q[11];
ry(1.509193479002421) q[10];
ry(-2.145215055830743) q[11];
cx q[10],q[11];
ry(-2.503184506224693) q[0];
ry(-0.543973035658721) q[2];
cx q[0],q[2];
ry(2.60416914518025) q[0];
ry(-1.8572574488594666) q[2];
cx q[0],q[2];
ry(0.04099362798440658) q[2];
ry(1.259580591919836) q[4];
cx q[2],q[4];
ry(-2.4457817465382385) q[2];
ry(-2.76186964259093) q[4];
cx q[2],q[4];
ry(1.2608809185101988) q[4];
ry(1.704769577037137) q[6];
cx q[4],q[6];
ry(0.44319057082167457) q[4];
ry(0.9957191546406702) q[6];
cx q[4],q[6];
ry(2.0826466673135466) q[6];
ry(0.21396789799970922) q[8];
cx q[6],q[8];
ry(-1.8402148768167814) q[6];
ry(1.6516031731124716) q[8];
cx q[6],q[8];
ry(-1.1259468968466937) q[8];
ry(-2.4175112893244193) q[10];
cx q[8],q[10];
ry(-2.2331714477198483) q[8];
ry(2.113125879528095) q[10];
cx q[8],q[10];
ry(0.8253900926375541) q[1];
ry(-0.16672867188907403) q[3];
cx q[1],q[3];
ry(0.17314742953484905) q[1];
ry(1.2564199673995682) q[3];
cx q[1],q[3];
ry(-0.5892938808411792) q[3];
ry(-2.8451691122977625) q[5];
cx q[3],q[5];
ry(-0.5173091174459845) q[3];
ry(1.230618776501994) q[5];
cx q[3],q[5];
ry(-2.614499616450115) q[5];
ry(-2.8718098792693323) q[7];
cx q[5],q[7];
ry(1.286697209285079) q[5];
ry(-0.3930441770505213) q[7];
cx q[5],q[7];
ry(1.7095754710335775) q[7];
ry(-0.17710568301425234) q[9];
cx q[7],q[9];
ry(0.7239549241929749) q[7];
ry(-0.46375588424175374) q[9];
cx q[7],q[9];
ry(-1.9778339416934723) q[9];
ry(2.8809107287945763) q[11];
cx q[9],q[11];
ry(0.4667858279183097) q[9];
ry(1.6497424426734737) q[11];
cx q[9],q[11];
ry(-3.0658406830057996) q[0];
ry(3.0486264004530237) q[1];
cx q[0],q[1];
ry(-0.8799402191217871) q[0];
ry(-0.6642806801291226) q[1];
cx q[0],q[1];
ry(-0.8703153139802067) q[2];
ry(1.1188645864000044) q[3];
cx q[2],q[3];
ry(-1.6470140462484806) q[2];
ry(-0.48604165157598184) q[3];
cx q[2],q[3];
ry(-2.8922610918939617) q[4];
ry(-2.922607411722718) q[5];
cx q[4],q[5];
ry(2.9743332403700253) q[4];
ry(0.7717291163752753) q[5];
cx q[4],q[5];
ry(-2.844179665497541) q[6];
ry(-2.9874778213420514) q[7];
cx q[6],q[7];
ry(-1.2652123921999854) q[6];
ry(0.5073233442466138) q[7];
cx q[6],q[7];
ry(2.1813442331102255) q[8];
ry(-0.5241501604494225) q[9];
cx q[8],q[9];
ry(1.514504978566567) q[8];
ry(-2.6311672750232056) q[9];
cx q[8],q[9];
ry(-0.785816779309809) q[10];
ry(1.0957011354358084) q[11];
cx q[10],q[11];
ry(1.0435680459511605) q[10];
ry(-1.7601678546160766) q[11];
cx q[10],q[11];
ry(-3.0841534120530563) q[0];
ry(2.796672823208038) q[2];
cx q[0],q[2];
ry(2.115347507184154) q[0];
ry(2.3152805526797278) q[2];
cx q[0],q[2];
ry(2.1767096695015575) q[2];
ry(2.4119532789802456) q[4];
cx q[2],q[4];
ry(0.2069676083903664) q[2];
ry(1.7503225911589464) q[4];
cx q[2],q[4];
ry(2.3576548753001685) q[4];
ry(1.0916300829614491) q[6];
cx q[4],q[6];
ry(-1.3021236330064878) q[4];
ry(1.6747287602452907) q[6];
cx q[4],q[6];
ry(1.1169000199542003) q[6];
ry(0.9034529737242104) q[8];
cx q[6],q[8];
ry(1.099437338032934) q[6];
ry(-0.4259571509856184) q[8];
cx q[6],q[8];
ry(-2.685982279789931) q[8];
ry(2.750383543735361) q[10];
cx q[8],q[10];
ry(-2.19681532258729) q[8];
ry(1.1552534304796636) q[10];
cx q[8],q[10];
ry(-2.1580986511575446) q[1];
ry(2.320104942287372) q[3];
cx q[1],q[3];
ry(0.4606882018050275) q[1];
ry(-1.5680619528683093) q[3];
cx q[1],q[3];
ry(-2.7944736680779774) q[3];
ry(2.1540916295542756) q[5];
cx q[3],q[5];
ry(2.9241920088925046) q[3];
ry(-0.2127658754911837) q[5];
cx q[3],q[5];
ry(1.5291167592303694) q[5];
ry(-0.03808432803909279) q[7];
cx q[5],q[7];
ry(0.572091080970817) q[5];
ry(1.9417856465589551) q[7];
cx q[5],q[7];
ry(-2.4775003510858693) q[7];
ry(-0.6733853400520082) q[9];
cx q[7],q[9];
ry(-1.228958996336506) q[7];
ry(0.6143615180667511) q[9];
cx q[7],q[9];
ry(0.8899609502670449) q[9];
ry(-1.5483794931034354) q[11];
cx q[9],q[11];
ry(-1.2932626854770277) q[9];
ry(-2.3141842137669526) q[11];
cx q[9],q[11];
ry(0.9972865454701764) q[0];
ry(1.674595776101734) q[1];
cx q[0],q[1];
ry(2.893721216621297) q[0];
ry(2.5484523407255977) q[1];
cx q[0],q[1];
ry(-1.8357499566836735) q[2];
ry(2.821294352653151) q[3];
cx q[2],q[3];
ry(1.2377624800907716) q[2];
ry(2.0145930046555467) q[3];
cx q[2],q[3];
ry(2.6473363408829678) q[4];
ry(1.9818725149283765) q[5];
cx q[4],q[5];
ry(-1.6627118002656724) q[4];
ry(1.7651059926851032) q[5];
cx q[4],q[5];
ry(-2.517772791798476) q[6];
ry(1.5319002381056013) q[7];
cx q[6],q[7];
ry(-0.6463669745264731) q[6];
ry(-1.9112934910749138) q[7];
cx q[6],q[7];
ry(0.45190382306148713) q[8];
ry(-1.151991846627686) q[9];
cx q[8],q[9];
ry(2.9684988597597655) q[8];
ry(2.727315328802201) q[9];
cx q[8],q[9];
ry(-0.6712943334702044) q[10];
ry(-1.43784365352077) q[11];
cx q[10],q[11];
ry(2.2373521935109064) q[10];
ry(-0.5809979758410825) q[11];
cx q[10],q[11];
ry(-0.3932745362251042) q[0];
ry(-1.8419875467042504) q[2];
cx q[0],q[2];
ry(0.14106888437175294) q[0];
ry(2.403958880489657) q[2];
cx q[0],q[2];
ry(2.8721475397525285) q[2];
ry(2.3221361754340726) q[4];
cx q[2],q[4];
ry(-0.9954531484731817) q[2];
ry(2.1496169366214364) q[4];
cx q[2],q[4];
ry(-1.9971003488785752) q[4];
ry(1.9735853397779057) q[6];
cx q[4],q[6];
ry(-2.8796108951160733) q[4];
ry(-0.5314754779305889) q[6];
cx q[4],q[6];
ry(-2.825754896578888) q[6];
ry(-2.3260615045872552) q[8];
cx q[6],q[8];
ry(-2.3003488131862717) q[6];
ry(0.4136880666786409) q[8];
cx q[6],q[8];
ry(-2.755285328077346) q[8];
ry(1.348936407613025) q[10];
cx q[8],q[10];
ry(2.5422071702441698) q[8];
ry(-1.3705196435683367) q[10];
cx q[8],q[10];
ry(-2.9948960960575093) q[1];
ry(-3.040201956594482) q[3];
cx q[1],q[3];
ry(-1.0280419719477418) q[1];
ry(-1.0529444791075067) q[3];
cx q[1],q[3];
ry(-2.6086330436244944) q[3];
ry(-0.2166945565204261) q[5];
cx q[3],q[5];
ry(2.021099382820791) q[3];
ry(2.4529708140336486) q[5];
cx q[3],q[5];
ry(-0.5070564512493937) q[5];
ry(-2.44654143894178) q[7];
cx q[5],q[7];
ry(-2.8783839186378204) q[5];
ry(1.7640445247787282) q[7];
cx q[5],q[7];
ry(0.9875401681681079) q[7];
ry(-0.6710928744675044) q[9];
cx q[7],q[9];
ry(2.194222840796919) q[7];
ry(1.6114073142911003) q[9];
cx q[7],q[9];
ry(2.826208076128897) q[9];
ry(-2.4599240044120743) q[11];
cx q[9],q[11];
ry(-2.6659506132829485) q[9];
ry(-1.8630617924742332) q[11];
cx q[9],q[11];
ry(2.5746610562543517) q[0];
ry(2.344942947988746) q[1];
cx q[0],q[1];
ry(2.227604817919434) q[0];
ry(3.0334775271384307) q[1];
cx q[0],q[1];
ry(-1.4593249862593778) q[2];
ry(-1.2990343302367808) q[3];
cx q[2],q[3];
ry(-0.5246396870026414) q[2];
ry(-1.840764076480042) q[3];
cx q[2],q[3];
ry(-0.8690892008891318) q[4];
ry(2.4980165570643984) q[5];
cx q[4],q[5];
ry(-0.531133055000522) q[4];
ry(2.718797895920691) q[5];
cx q[4],q[5];
ry(1.9743321552814679) q[6];
ry(-1.5257158139075553) q[7];
cx q[6],q[7];
ry(-1.8031459065175257) q[6];
ry(-2.1402362441732627) q[7];
cx q[6],q[7];
ry(-2.0927574194371337) q[8];
ry(-1.0366019583730006) q[9];
cx q[8],q[9];
ry(1.7375495419022056) q[8];
ry(0.9836840927276702) q[9];
cx q[8],q[9];
ry(0.7898059576616898) q[10];
ry(-1.9220358722881958) q[11];
cx q[10],q[11];
ry(-1.8451456026225026) q[10];
ry(-2.086663394655681) q[11];
cx q[10],q[11];
ry(-0.2780776339018113) q[0];
ry(0.7284880033220711) q[2];
cx q[0],q[2];
ry(-2.89930317326265) q[0];
ry(0.8015908973868635) q[2];
cx q[0],q[2];
ry(-1.6895577435884723) q[2];
ry(2.6166063567653204) q[4];
cx q[2],q[4];
ry(-1.0327289563732105) q[2];
ry(-1.9335061975418844) q[4];
cx q[2],q[4];
ry(0.8053247124993977) q[4];
ry(0.2641779315704911) q[6];
cx q[4],q[6];
ry(-2.028481119126842) q[4];
ry(0.22200071925377873) q[6];
cx q[4],q[6];
ry(2.9794522057756017) q[6];
ry(1.3304999800118287) q[8];
cx q[6],q[8];
ry(2.7778123353740383) q[6];
ry(2.646071486538322) q[8];
cx q[6],q[8];
ry(-2.116263231098272) q[8];
ry(-1.7508620373451347) q[10];
cx q[8],q[10];
ry(-2.989527237825516) q[8];
ry(0.9953166979540835) q[10];
cx q[8],q[10];
ry(1.6484107104811967) q[1];
ry(2.7715295993822857) q[3];
cx q[1],q[3];
ry(2.5908764897210257) q[1];
ry(-1.1726005900326675) q[3];
cx q[1],q[3];
ry(-2.9630077092303773) q[3];
ry(-1.3049123106989438) q[5];
cx q[3],q[5];
ry(-1.752932844198279) q[3];
ry(-2.673290469350929) q[5];
cx q[3],q[5];
ry(-0.5376759929066157) q[5];
ry(1.2591514107749076) q[7];
cx q[5],q[7];
ry(1.099965311295849) q[5];
ry(2.5514675728133156) q[7];
cx q[5],q[7];
ry(1.5766249917794617) q[7];
ry(-1.8930761621016237) q[9];
cx q[7],q[9];
ry(0.36776596628001723) q[7];
ry(-3.0307953895543114) q[9];
cx q[7],q[9];
ry(1.1167523448403391) q[9];
ry(-2.2215088091230415) q[11];
cx q[9],q[11];
ry(0.6757763596310211) q[9];
ry(1.8338766124378423) q[11];
cx q[9],q[11];
ry(-2.1914112816583646) q[0];
ry(0.03196587881714559) q[1];
cx q[0],q[1];
ry(0.9501219649978528) q[0];
ry(0.23167682735519968) q[1];
cx q[0],q[1];
ry(2.7266681344374586) q[2];
ry(-2.73434887534598) q[3];
cx q[2],q[3];
ry(-0.398886944804171) q[2];
ry(-2.28665899183516) q[3];
cx q[2],q[3];
ry(-2.0921975929373193) q[4];
ry(-0.7436768979720147) q[5];
cx q[4],q[5];
ry(-1.0124216129152785) q[4];
ry(1.4656285804886435) q[5];
cx q[4],q[5];
ry(-0.3209921351945262) q[6];
ry(2.4330066893379447) q[7];
cx q[6],q[7];
ry(2.643195888237993) q[6];
ry(-0.5155604671544162) q[7];
cx q[6],q[7];
ry(-0.16400978517529233) q[8];
ry(-1.0610627678528841) q[9];
cx q[8],q[9];
ry(-1.278812267134227) q[8];
ry(2.2232294712235907) q[9];
cx q[8],q[9];
ry(2.671710006412714) q[10];
ry(2.8022797309757927) q[11];
cx q[10],q[11];
ry(-0.266991269053065) q[10];
ry(-0.4507035430060249) q[11];
cx q[10],q[11];
ry(-0.49419904798942965) q[0];
ry(1.2195196353807338) q[2];
cx q[0],q[2];
ry(-1.874931000525479) q[0];
ry(-0.7816712944376834) q[2];
cx q[0],q[2];
ry(-0.12846416219449086) q[2];
ry(0.7028139308758163) q[4];
cx q[2],q[4];
ry(1.0360085392643992) q[2];
ry(-0.8339869134662584) q[4];
cx q[2],q[4];
ry(-0.508765216736923) q[4];
ry(0.35815402056758233) q[6];
cx q[4],q[6];
ry(-0.46052851310200804) q[4];
ry(2.7498688277465506) q[6];
cx q[4],q[6];
ry(0.8339156514044674) q[6];
ry(-2.665081676312731) q[8];
cx q[6],q[8];
ry(-1.1697153301662133) q[6];
ry(1.5113296353464882) q[8];
cx q[6],q[8];
ry(-0.23202030242720273) q[8];
ry(2.680148644234724) q[10];
cx q[8],q[10];
ry(0.9697397425108756) q[8];
ry(-0.9156366040183391) q[10];
cx q[8],q[10];
ry(1.8363146072284426) q[1];
ry(-2.6751714189780667) q[3];
cx q[1],q[3];
ry(-2.019748175035076) q[1];
ry(2.70837989859692) q[3];
cx q[1],q[3];
ry(-2.3708093279765867) q[3];
ry(1.1046667957955139) q[5];
cx q[3],q[5];
ry(1.164894335238944) q[3];
ry(1.2709323129888936) q[5];
cx q[3],q[5];
ry(-1.6185713713792929) q[5];
ry(2.618631068919664) q[7];
cx q[5],q[7];
ry(1.1904128625092225) q[5];
ry(-1.8713516127605112) q[7];
cx q[5],q[7];
ry(1.0690347754945666) q[7];
ry(2.759468395537982) q[9];
cx q[7],q[9];
ry(2.251121725467271) q[7];
ry(0.5378977908989402) q[9];
cx q[7],q[9];
ry(1.945267339293241) q[9];
ry(-0.925545358591655) q[11];
cx q[9],q[11];
ry(2.455031753552824) q[9];
ry(-1.9060294184917315) q[11];
cx q[9],q[11];
ry(-2.761896884614392) q[0];
ry(-0.8249563690759318) q[1];
cx q[0],q[1];
ry(-0.8473392198961616) q[0];
ry(-1.5768613342098474) q[1];
cx q[0],q[1];
ry(-0.8845997902055799) q[2];
ry(-0.05843537667688645) q[3];
cx q[2],q[3];
ry(0.8241438242091421) q[2];
ry(2.261867952894825) q[3];
cx q[2],q[3];
ry(0.36355639064426737) q[4];
ry(-0.9184524740081823) q[5];
cx q[4],q[5];
ry(-0.9428850031392296) q[4];
ry(2.0076713633488534) q[5];
cx q[4],q[5];
ry(-2.2453848262274754) q[6];
ry(2.8771798574965874) q[7];
cx q[6],q[7];
ry(2.0459229710433675) q[6];
ry(2.2494045341943325) q[7];
cx q[6],q[7];
ry(-2.6037979201293786) q[8];
ry(-2.6001924471454427) q[9];
cx q[8],q[9];
ry(-0.2628379864266339) q[8];
ry(1.5004462568060817) q[9];
cx q[8],q[9];
ry(2.2940905520311103) q[10];
ry(-2.9765203534198608) q[11];
cx q[10],q[11];
ry(-2.3227574254458783) q[10];
ry(2.139578654827468) q[11];
cx q[10],q[11];
ry(-1.2258589626238743) q[0];
ry(-2.486355901785857) q[2];
cx q[0],q[2];
ry(0.5654218169668742) q[0];
ry(-2.806150305421106) q[2];
cx q[0],q[2];
ry(1.868766893128619) q[2];
ry(-1.9335038863588485) q[4];
cx q[2],q[4];
ry(2.7835223014517796) q[2];
ry(0.8935407737128913) q[4];
cx q[2],q[4];
ry(-0.7821497925952174) q[4];
ry(-2.832279078253803) q[6];
cx q[4],q[6];
ry(-1.9666475927963702) q[4];
ry(-2.1098609370657737) q[6];
cx q[4],q[6];
ry(0.9110162959135231) q[6];
ry(-1.9217914473841669) q[8];
cx q[6],q[8];
ry(-0.30327621386449893) q[6];
ry(-2.616045451805933) q[8];
cx q[6],q[8];
ry(-1.0527468114449239) q[8];
ry(1.621494935602405) q[10];
cx q[8],q[10];
ry(2.9710826569330178) q[8];
ry(1.8448526262597844) q[10];
cx q[8],q[10];
ry(0.5827739225987463) q[1];
ry(1.1530954587321884) q[3];
cx q[1],q[3];
ry(2.949421453673723) q[1];
ry(-2.015594663746179) q[3];
cx q[1],q[3];
ry(1.0581108943105715) q[3];
ry(-0.21782512280214486) q[5];
cx q[3],q[5];
ry(-1.990747261366672) q[3];
ry(-0.9318539888549191) q[5];
cx q[3],q[5];
ry(1.8286517189593994) q[5];
ry(0.12211639102153704) q[7];
cx q[5],q[7];
ry(-1.6724589229576907) q[5];
ry(-0.40478465556122506) q[7];
cx q[5],q[7];
ry(-3.0942996080110907) q[7];
ry(0.9674658030925245) q[9];
cx q[7],q[9];
ry(1.505716265828953) q[7];
ry(2.5486606878938636) q[9];
cx q[7],q[9];
ry(0.5756772701206678) q[9];
ry(-2.4789413457163882) q[11];
cx q[9],q[11];
ry(-2.081163926191476) q[9];
ry(1.7710505432759165) q[11];
cx q[9],q[11];
ry(2.8267512394897283) q[0];
ry(-1.4878743233491711) q[1];
ry(-3.0241212285110093) q[2];
ry(2.1278790419245586) q[3];
ry(0.03609221963372455) q[4];
ry(0.7214701810675798) q[5];
ry(-3.1033349089750413) q[6];
ry(2.3348216934998014) q[7];
ry(0.09374077883460785) q[8];
ry(-0.7477496635362434) q[9];
ry(0.32526848507685013) q[10];
ry(0.4290633500989667) q[11];