OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.8550717325512625) q[0];
ry(0.9297594974640369) q[1];
cx q[0],q[1];
ry(-1.1558899368560998) q[0];
ry(1.7112762182019947) q[1];
cx q[0],q[1];
ry(2.6090461740444253) q[2];
ry(-1.0295409033104628) q[3];
cx q[2],q[3];
ry(1.9539011453040043) q[2];
ry(2.950931229811842) q[3];
cx q[2],q[3];
ry(3.040811678848942) q[4];
ry(-2.9785529765660073) q[5];
cx q[4],q[5];
ry(-1.3717372143652016) q[4];
ry(1.723365998915195) q[5];
cx q[4],q[5];
ry(2.297671860468582) q[6];
ry(2.3707525413405293) q[7];
cx q[6],q[7];
ry(-2.394996895656941) q[6];
ry(-1.0162226828556333) q[7];
cx q[6],q[7];
ry(1.9644621263525632) q[8];
ry(-1.355736724360896) q[9];
cx q[8],q[9];
ry(-1.7312813438714327) q[8];
ry(0.5103302103022243) q[9];
cx q[8],q[9];
ry(-0.13236366495581822) q[10];
ry(0.33960655811196805) q[11];
cx q[10],q[11];
ry(0.2942001974884947) q[10];
ry(-1.301004796496722) q[11];
cx q[10],q[11];
ry(-2.624662264530687) q[12];
ry(2.444013533887552) q[13];
cx q[12],q[13];
ry(-2.3788876294865067) q[12];
ry(0.6237701426808178) q[13];
cx q[12],q[13];
ry(-0.6598250127316314) q[14];
ry(0.07220101647124014) q[15];
cx q[14],q[15];
ry(-0.5009344540287459) q[14];
ry(-2.47035225182932) q[15];
cx q[14],q[15];
ry(0.658262359138999) q[16];
ry(1.3106853499513165) q[17];
cx q[16],q[17];
ry(-0.5011120985608739) q[16];
ry(-1.74556038627885) q[17];
cx q[16],q[17];
ry(-0.11428227626771585) q[18];
ry(-2.6005053835172682) q[19];
cx q[18],q[19];
ry(2.4451675679310307) q[18];
ry(2.7011040895206193) q[19];
cx q[18],q[19];
ry(-1.6370620402233693) q[1];
ry(-3.0767260179056892) q[2];
cx q[1],q[2];
ry(-2.991348904343439) q[1];
ry(0.2521408797099453) q[2];
cx q[1],q[2];
ry(2.844137618277236) q[3];
ry(-0.011640429684487941) q[4];
cx q[3],q[4];
ry(1.3752524317258752) q[3];
ry(-2.894647460050297) q[4];
cx q[3],q[4];
ry(-0.08774879696563802) q[5];
ry(-2.7245696586610415) q[6];
cx q[5],q[6];
ry(0.3562751324936819) q[5];
ry(-1.3796938119660163) q[6];
cx q[5],q[6];
ry(0.6632563987260314) q[7];
ry(-2.7876415248225395) q[8];
cx q[7],q[8];
ry(0.9624635673940345) q[7];
ry(0.500866033725889) q[8];
cx q[7],q[8];
ry(-1.4256100696960694) q[9];
ry(0.06867952575016238) q[10];
cx q[9],q[10];
ry(1.673648133052071) q[9];
ry(0.388317294248257) q[10];
cx q[9],q[10];
ry(1.0860444675334353) q[11];
ry(-0.31321497540529764) q[12];
cx q[11],q[12];
ry(0.9051511522236227) q[11];
ry(0.6878298532808778) q[12];
cx q[11],q[12];
ry(0.7045964174059529) q[13];
ry(1.5686708103800995) q[14];
cx q[13],q[14];
ry(1.864411639584463) q[13];
ry(-1.989944587766109) q[14];
cx q[13],q[14];
ry(1.2557393271522257) q[15];
ry(-2.296255928134564) q[16];
cx q[15],q[16];
ry(1.263619227749465) q[15];
ry(1.088558258943075) q[16];
cx q[15],q[16];
ry(-0.3197311490057478) q[17];
ry(-0.1732511937604828) q[18];
cx q[17],q[18];
ry(0.9892225250665487) q[17];
ry(0.008072957975780781) q[18];
cx q[17],q[18];
ry(2.836827832343056) q[0];
ry(-3.023563250036216) q[1];
cx q[0],q[1];
ry(-1.6796106663075694) q[0];
ry(0.029162055601542888) q[1];
cx q[0],q[1];
ry(2.4219686485709846) q[2];
ry(-1.0255722556384423) q[3];
cx q[2],q[3];
ry(-1.1223076192230605) q[2];
ry(-2.159893469032659) q[3];
cx q[2],q[3];
ry(3.0495237726492888) q[4];
ry(-1.17155883975955) q[5];
cx q[4],q[5];
ry(-2.7346922264953037) q[4];
ry(1.6838879007125418) q[5];
cx q[4],q[5];
ry(0.39632082168800264) q[6];
ry(-0.6574304834221787) q[7];
cx q[6],q[7];
ry(2.8142894485785757) q[6];
ry(3.1101903442948142) q[7];
cx q[6],q[7];
ry(0.2482501627106066) q[8];
ry(2.3236682518764997) q[9];
cx q[8],q[9];
ry(-2.1118922118671057) q[8];
ry(2.4392722870248407) q[9];
cx q[8],q[9];
ry(-0.7089487706647937) q[10];
ry(2.8005224154719466) q[11];
cx q[10],q[11];
ry(-1.9710666996621518) q[10];
ry(-0.09686178378460092) q[11];
cx q[10],q[11];
ry(1.9764641780645702) q[12];
ry(0.03616850934536142) q[13];
cx q[12],q[13];
ry(-2.277466257605666) q[12];
ry(-0.6746788002780447) q[13];
cx q[12],q[13];
ry(2.6434760870246796) q[14];
ry(-1.4168476878305651) q[15];
cx q[14],q[15];
ry(0.39744162130335425) q[14];
ry(2.7071439717567696) q[15];
cx q[14],q[15];
ry(-2.3404556636865714) q[16];
ry(-0.40715386050522806) q[17];
cx q[16],q[17];
ry(-1.2041378655256185) q[16];
ry(-1.9216270655460905) q[17];
cx q[16],q[17];
ry(0.8539462265405702) q[18];
ry(-2.619011821190933) q[19];
cx q[18],q[19];
ry(2.3786404622330664) q[18];
ry(-0.4224549571202143) q[19];
cx q[18],q[19];
ry(-2.40263509746426) q[1];
ry(2.5127236300390345) q[2];
cx q[1],q[2];
ry(2.8659296208914533) q[1];
ry(0.2774581585433521) q[2];
cx q[1],q[2];
ry(2.5687611044329834) q[3];
ry(-2.8997518832160183) q[4];
cx q[3],q[4];
ry(-0.028166992685005) q[3];
ry(0.8485690458110642) q[4];
cx q[3],q[4];
ry(0.16734861284757052) q[5];
ry(-2.1688568201635667) q[6];
cx q[5],q[6];
ry(-0.15403090268702077) q[5];
ry(2.950426677209039) q[6];
cx q[5],q[6];
ry(0.28666016010759154) q[7];
ry(2.1465154657888887) q[8];
cx q[7],q[8];
ry(3.0911423368154827) q[7];
ry(-1.61781728628561) q[8];
cx q[7],q[8];
ry(1.356315580954807) q[9];
ry(1.4666545010198415) q[10];
cx q[9],q[10];
ry(-3.008243452305057) q[9];
ry(-0.9547947641696828) q[10];
cx q[9],q[10];
ry(2.91976020035656) q[11];
ry(0.8554361491640591) q[12];
cx q[11],q[12];
ry(2.965772819428123) q[11];
ry(-0.055588567224316605) q[12];
cx q[11],q[12];
ry(-1.5769029479674885) q[13];
ry(2.589736909956027) q[14];
cx q[13],q[14];
ry(-0.3989549180343987) q[13];
ry(-3.0833711686821643) q[14];
cx q[13],q[14];
ry(1.9090104330392093) q[15];
ry(1.9850456377726218) q[16];
cx q[15],q[16];
ry(3.128750155537672) q[15];
ry(1.4218429074760142) q[16];
cx q[15],q[16];
ry(-3.13346402634651) q[17];
ry(-0.2787240489858682) q[18];
cx q[17],q[18];
ry(2.9320719386201555) q[17];
ry(-3.075297857426766) q[18];
cx q[17],q[18];
ry(-0.3169485240194003) q[0];
ry(-0.7548979583098641) q[1];
cx q[0],q[1];
ry(-3.0086621485082468) q[0];
ry(-2.5346089934062466) q[1];
cx q[0],q[1];
ry(0.7654416054634947) q[2];
ry(2.9956442809616983) q[3];
cx q[2],q[3];
ry(-2.0650394651046895) q[2];
ry(-2.7586829578925247) q[3];
cx q[2],q[3];
ry(1.6355755163243506) q[4];
ry(-1.8689562433595723) q[5];
cx q[4],q[5];
ry(0.23245204212760623) q[4];
ry(-0.012041961364292142) q[5];
cx q[4],q[5];
ry(1.844272604878589) q[6];
ry(0.4360393635803863) q[7];
cx q[6],q[7];
ry(-1.1421286953895253) q[6];
ry(2.9578368552281042) q[7];
cx q[6],q[7];
ry(-2.75545680075343) q[8];
ry(2.2641798961936814) q[9];
cx q[8],q[9];
ry(-2.563554705463021) q[8];
ry(0.12479779407381031) q[9];
cx q[8],q[9];
ry(-0.6010140783028992) q[10];
ry(1.125598646701835) q[11];
cx q[10],q[11];
ry(-0.15073639769499714) q[10];
ry(2.716762621545205) q[11];
cx q[10],q[11];
ry(-3.057759072565668) q[12];
ry(0.6891974081039116) q[13];
cx q[12],q[13];
ry(0.19008810425772746) q[12];
ry(0.9290983006904908) q[13];
cx q[12],q[13];
ry(-2.1908104387441885) q[14];
ry(0.9163208530127881) q[15];
cx q[14],q[15];
ry(0.6261746306531408) q[14];
ry(-1.4680537341283744) q[15];
cx q[14],q[15];
ry(1.883678312751396) q[16];
ry(2.403214644157371) q[17];
cx q[16],q[17];
ry(0.2365523542764105) q[16];
ry(2.909519522688588) q[17];
cx q[16],q[17];
ry(0.928900328824663) q[18];
ry(-1.4747138769988317) q[19];
cx q[18],q[19];
ry(-1.6954862153134824) q[18];
ry(1.1968584025747058) q[19];
cx q[18],q[19];
ry(2.837690898520237) q[1];
ry(-0.033522306983979455) q[2];
cx q[1],q[2];
ry(2.596937700511354) q[1];
ry(2.5605225994166942) q[2];
cx q[1],q[2];
ry(-1.7001719361914205) q[3];
ry(0.9011717157674974) q[4];
cx q[3],q[4];
ry(-0.021570072344046842) q[3];
ry(1.582733034711401) q[4];
cx q[3],q[4];
ry(0.19638180894884924) q[5];
ry(0.13376278556152252) q[6];
cx q[5],q[6];
ry(3.0891170889316055) q[5];
ry(-0.17212268103971629) q[6];
cx q[5],q[6];
ry(-1.9437201141131304) q[7];
ry(-2.4787059842597743) q[8];
cx q[7],q[8];
ry(3.134095472719882) q[7];
ry(0.9975390960683115) q[8];
cx q[7],q[8];
ry(2.545622111851125) q[9];
ry(-2.799236631167016) q[10];
cx q[9],q[10];
ry(-0.04074094633145986) q[9];
ry(-0.05072465087712618) q[10];
cx q[9],q[10];
ry(-1.3830204016889514) q[11];
ry(2.256528395787875) q[12];
cx q[11],q[12];
ry(-1.2476387982652826) q[11];
ry(1.7816095527517346) q[12];
cx q[11],q[12];
ry(2.8958662460821283) q[13];
ry(0.06732026391830814) q[14];
cx q[13],q[14];
ry(-2.109807892203767) q[13];
ry(-0.01261832438302779) q[14];
cx q[13],q[14];
ry(2.5508750758284076) q[15];
ry(1.5641536208743085) q[16];
cx q[15],q[16];
ry(-3.127747289977071) q[15];
ry(-0.1797043673254178) q[16];
cx q[15],q[16];
ry(-2.016174960384869) q[17];
ry(1.3373397697620897) q[18];
cx q[17],q[18];
ry(1.8515446740762762) q[17];
ry(-0.27921981741339685) q[18];
cx q[17],q[18];
ry(-0.5128026547158202) q[0];
ry(2.251095744878774) q[1];
cx q[0],q[1];
ry(-1.991085009453597) q[0];
ry(1.0808813019149621) q[1];
cx q[0],q[1];
ry(2.482303090946741) q[2];
ry(0.5792498800015579) q[3];
cx q[2],q[3];
ry(2.1876159318872648) q[2];
ry(1.351566078261909) q[3];
cx q[2],q[3];
ry(0.5552033875893307) q[4];
ry(2.401219991823874) q[5];
cx q[4],q[5];
ry(-0.829815785473917) q[4];
ry(0.3994846767093486) q[5];
cx q[4],q[5];
ry(-2.778638703936651) q[6];
ry(-1.1669766885183055) q[7];
cx q[6],q[7];
ry(-1.9103843437868802) q[6];
ry(1.7178882773494246) q[7];
cx q[6],q[7];
ry(1.218995078692906) q[8];
ry(0.0028142703476105044) q[9];
cx q[8],q[9];
ry(1.1874624327370285) q[8];
ry(-1.5725719107429175) q[9];
cx q[8],q[9];
ry(-2.8291643296205065) q[10];
ry(-1.594840873024082) q[11];
cx q[10],q[11];
ry(1.3711903330359503) q[10];
ry(3.1081052277998924) q[11];
cx q[10],q[11];
ry(-0.7921580725344659) q[12];
ry(-1.647608387746556) q[13];
cx q[12],q[13];
ry(3.134908537182496) q[12];
ry(-0.3704950013681314) q[13];
cx q[12],q[13];
ry(1.40481730288686) q[14];
ry(-2.3848878303875494) q[15];
cx q[14],q[15];
ry(-1.4129435997184947) q[14];
ry(1.6225071800394864) q[15];
cx q[14],q[15];
ry(3.0287006366581304) q[16];
ry(2.7105286841031897) q[17];
cx q[16],q[17];
ry(1.8624226273535038) q[16];
ry(-2.906426968258605) q[17];
cx q[16],q[17];
ry(-0.4915707188753827) q[18];
ry(0.9049567237689934) q[19];
cx q[18],q[19];
ry(-1.5139382211288706) q[18];
ry(0.7055379053637056) q[19];
cx q[18],q[19];
ry(0.8194521913773648) q[1];
ry(-3.0900271857182346) q[2];
cx q[1],q[2];
ry(3.0858023130571572) q[1];
ry(0.7438381615674273) q[2];
cx q[1],q[2];
ry(3.1171360733166016) q[3];
ry(-1.0952064510012303) q[4];
cx q[3],q[4];
ry(-3.1174097231068676) q[3];
ry(-0.025489470110725776) q[4];
cx q[3],q[4];
ry(-1.7131395307961523) q[5];
ry(-0.5996491609148481) q[6];
cx q[5],q[6];
ry(-2.7964355274261914) q[5];
ry(-0.17271037030383077) q[6];
cx q[5],q[6];
ry(2.092822348078582) q[7];
ry(-1.5314786318450535) q[8];
cx q[7],q[8];
ry(-1.9118735879964317) q[7];
ry(0.6569517602390906) q[8];
cx q[7],q[8];
ry(-0.07942773891361599) q[9];
ry(1.9027489373613031) q[10];
cx q[9],q[10];
ry(-1.5961642070550983) q[9];
ry(-3.133128073453581) q[10];
cx q[9],q[10];
ry(1.600520361845934) q[11];
ry(-1.1915348904307095) q[12];
cx q[11],q[12];
ry(3.124584926352096) q[11];
ry(-2.565258547845644) q[12];
cx q[11],q[12];
ry(-0.5183582791050112) q[13];
ry(-1.4972022790668982) q[14];
cx q[13],q[14];
ry(1.9335205693567818) q[13];
ry(3.1333605469130523) q[14];
cx q[13],q[14];
ry(1.2422096982812398) q[15];
ry(0.183111901208147) q[16];
cx q[15],q[16];
ry(-2.1582144541925032) q[15];
ry(1.6171994285974995) q[16];
cx q[15],q[16];
ry(1.6253509852486092) q[17];
ry(-0.48395448491980164) q[18];
cx q[17],q[18];
ry(0.6083861067398619) q[17];
ry(1.6345149062738669) q[18];
cx q[17],q[18];
ry(-2.9234375374071124) q[0];
ry(1.9219190889320497) q[1];
cx q[0],q[1];
ry(1.1378375878455431) q[0];
ry(-0.687299675453775) q[1];
cx q[0],q[1];
ry(0.3849719529462979) q[2];
ry(0.9873896686895955) q[3];
cx q[2],q[3];
ry(0.5576651511660576) q[2];
ry(-2.1364857173938683) q[3];
cx q[2],q[3];
ry(-2.739245393324269) q[4];
ry(2.0670179986563) q[5];
cx q[4],q[5];
ry(-3.1259469474573507) q[4];
ry(0.7033109256606531) q[5];
cx q[4],q[5];
ry(0.8748418651530726) q[6];
ry(-1.2138992490815241) q[7];
cx q[6],q[7];
ry(0.02536975592917776) q[6];
ry(-0.00015891891793540228) q[7];
cx q[6],q[7];
ry(-1.594451812783892) q[8];
ry(3.077257321525432) q[9];
cx q[8],q[9];
ry(-1.4156826756003222) q[8];
ry(-1.8333240630257324) q[9];
cx q[8],q[9];
ry(1.5631390040649802) q[10];
ry(2.9574693237881737) q[11];
cx q[10],q[11];
ry(0.003970462449866447) q[10];
ry(-1.075592097349563) q[11];
cx q[10],q[11];
ry(0.4899255746931031) q[12];
ry(2.5645187424426465) q[13];
cx q[12],q[13];
ry(-2.138543849207675) q[12];
ry(-2.781999227063955) q[13];
cx q[12],q[13];
ry(1.6794554166927655) q[14];
ry(1.2169552741577618) q[15];
cx q[14],q[15];
ry(2.8484471850808304) q[14];
ry(0.7931885681150135) q[15];
cx q[14],q[15];
ry(-1.6526084822043803) q[16];
ry(-0.4085705741226448) q[17];
cx q[16],q[17];
ry(1.6534054369192535) q[16];
ry(0.3272491859718638) q[17];
cx q[16],q[17];
ry(-0.7743463014519558) q[18];
ry(-0.6231824838047482) q[19];
cx q[18],q[19];
ry(1.4883432587910796) q[18];
ry(-1.117344947407066) q[19];
cx q[18],q[19];
ry(0.834502280754545) q[1];
ry(-1.9609642121191708) q[2];
cx q[1],q[2];
ry(0.01781262937830136) q[1];
ry(0.8809835960792233) q[2];
cx q[1],q[2];
ry(1.8534803601833891) q[3];
ry(3.042451412605924) q[4];
cx q[3],q[4];
ry(-3.13951956472219) q[3];
ry(-3.068636607124223) q[4];
cx q[3],q[4];
ry(-0.8312157245192073) q[5];
ry(0.5746512747319255) q[6];
cx q[5],q[6];
ry(-0.27680319590945496) q[5];
ry(2.053791687676868) q[6];
cx q[5],q[6];
ry(0.8928204550914137) q[7];
ry(-2.2482871131053628) q[8];
cx q[7],q[8];
ry(-3.14070197061316) q[7];
ry(-0.7415167590528515) q[8];
cx q[7],q[8];
ry(0.7269699624228974) q[9];
ry(1.5690781594160625) q[10];
cx q[9],q[10];
ry(1.4808558331668147) q[9];
ry(-1.9566232645862915) q[10];
cx q[9],q[10];
ry(-2.9093566689697172) q[11];
ry(2.5664672025901747) q[12];
cx q[11],q[12];
ry(0.3251174295778977) q[11];
ry(1.1932488864896393) q[12];
cx q[11],q[12];
ry(-1.578371692127499) q[13];
ry(-0.051574738224509684) q[14];
cx q[13],q[14];
ry(0.008407457095598756) q[13];
ry(-0.9271135297352077) q[14];
cx q[13],q[14];
ry(-1.2649410136841697) q[15];
ry(-2.998224327798278) q[16];
cx q[15],q[16];
ry(-0.2585770613701251) q[15];
ry(3.08548513662684) q[16];
cx q[15],q[16];
ry(-0.6060061615945079) q[17];
ry(-1.9052345808996456) q[18];
cx q[17],q[18];
ry(0.2134782212429558) q[17];
ry(0.897565017756861) q[18];
cx q[17],q[18];
ry(-0.4130316941477856) q[0];
ry(-2.244199830120089) q[1];
cx q[0],q[1];
ry(-0.5876195013924431) q[0];
ry(1.4544093782948277) q[1];
cx q[0],q[1];
ry(-1.2116341715377317) q[2];
ry(-3.1083239297411724) q[3];
cx q[2],q[3];
ry(0.2862769489861883) q[2];
ry(-1.5175316523974411) q[3];
cx q[2],q[3];
ry(-2.607661111343278) q[4];
ry(-1.4697907603075242) q[5];
cx q[4],q[5];
ry(-1.213789155382134) q[4];
ry(1.7363916675783158) q[5];
cx q[4],q[5];
ry(0.8722446519962052) q[6];
ry(1.8929665672733746) q[7];
cx q[6],q[7];
ry(-1.5688688165312163) q[6];
ry(1.608758521985047) q[7];
cx q[6],q[7];
ry(-1.989517409594061) q[8];
ry(-1.0543781146711189) q[9];
cx q[8],q[9];
ry(3.133521864571975) q[8];
ry(-3.141244466014275) q[9];
cx q[8],q[9];
ry(-1.737393963606681) q[10];
ry(1.6002070148629421) q[11];
cx q[10],q[11];
ry(-0.7043511934997335) q[10];
ry(-0.0008162724144674029) q[11];
cx q[10],q[11];
ry(1.5539197769112902) q[12];
ry(1.6699494470851084) q[13];
cx q[12],q[13];
ry(-3.135041924924308) q[12];
ry(0.9661756082410736) q[13];
cx q[12],q[13];
ry(0.10835105907358011) q[14];
ry(2.558148820635447) q[15];
cx q[14],q[15];
ry(2.918167666491284) q[14];
ry(-1.9211856665220868) q[15];
cx q[14],q[15];
ry(-1.8440057720083365) q[16];
ry(0.3586726473236764) q[17];
cx q[16],q[17];
ry(0.0646945171243507) q[16];
ry(-3.1196790933267535) q[17];
cx q[16],q[17];
ry(0.1844353132108854) q[18];
ry(-1.437853105525189) q[19];
cx q[18],q[19];
ry(-2.1479366222793463) q[18];
ry(0.18494414092562028) q[19];
cx q[18],q[19];
ry(-0.7639442763740887) q[1];
ry(-1.4487570118720692) q[2];
cx q[1],q[2];
ry(0.7097316246826697) q[1];
ry(0.1818304766648901) q[2];
cx q[1],q[2];
ry(1.6029067235193377) q[3];
ry(1.579347032970138) q[4];
cx q[3],q[4];
ry(-1.8639192196859318) q[3];
ry(-1.6774777422961715) q[4];
cx q[3],q[4];
ry(-1.657090260952648) q[5];
ry(-1.6707575316000833) q[6];
cx q[5],q[6];
ry(-3.005928122864944) q[5];
ry(-0.28103708239915465) q[6];
cx q[5],q[6];
ry(-1.565210394873465) q[7];
ry(1.3777771011107587) q[8];
cx q[7],q[8];
ry(-0.0030849398141851534) q[7];
ry(2.8443083305422436) q[8];
cx q[7],q[8];
ry(-3.0644210743336546) q[9];
ry(-1.3876689405589089) q[10];
cx q[9],q[10];
ry(0.24644971617909034) q[9];
ry(-0.961342019355574) q[10];
cx q[9],q[10];
ry(1.569211910420024) q[11];
ry(-1.5725284923989349) q[12];
cx q[11],q[12];
ry(-0.8071826667546225) q[11];
ry(-1.2592705273945903) q[12];
cx q[11],q[12];
ry(1.6634252556708888) q[13];
ry(0.6808647576761739) q[14];
cx q[13],q[14];
ry(3.1378684505919656) q[13];
ry(0.2688551760158093) q[14];
cx q[13],q[14];
ry(2.3159459595510783) q[15];
ry(-3.139764875751979) q[16];
cx q[15],q[16];
ry(3.02927714262738) q[15];
ry(0.07249607151109405) q[16];
cx q[15],q[16];
ry(2.8770780261133124) q[17];
ry(2.3418209843994666) q[18];
cx q[17],q[18];
ry(0.03760701385304977) q[17];
ry(2.0119575240307057) q[18];
cx q[17],q[18];
ry(2.6819960276232906) q[0];
ry(2.1553576801510475) q[1];
cx q[0],q[1];
ry(-1.1973015502077855) q[0];
ry(-1.256138702692397) q[1];
cx q[0],q[1];
ry(-1.025803267619514) q[2];
ry(2.6331358814347925) q[3];
cx q[2],q[3];
ry(-2.198964034191585) q[2];
ry(2.2248422497354428) q[3];
cx q[2],q[3];
ry(1.4720343886646294) q[4];
ry(-0.9254320273637768) q[5];
cx q[4],q[5];
ry(-0.03735124234256416) q[4];
ry(-0.21862774513162606) q[5];
cx q[4],q[5];
ry(-1.671672621934542) q[6];
ry(1.3305661318393014) q[7];
cx q[6],q[7];
ry(3.125064783944157) q[6];
ry(1.0044137638087296) q[7];
cx q[6],q[7];
ry(-1.1108732248951858) q[8];
ry(0.6753942302283764) q[9];
cx q[8],q[9];
ry(0.03738135098290069) q[8];
ry(1.0226838777100138) q[9];
cx q[8],q[9];
ry(1.5531184998391019) q[10];
ry(-1.5730147801688166) q[11];
cx q[10],q[11];
ry(2.092541277636838) q[10];
ry(1.2911810352949018) q[11];
cx q[10],q[11];
ry(2.927464388079492) q[12];
ry(-1.5733947884747241) q[13];
cx q[12],q[13];
ry(2.203024139742828) q[12];
ry(-0.0012396656004520776) q[13];
cx q[12],q[13];
ry(0.6509610526587043) q[14];
ry(1.4889086442523298) q[15];
cx q[14],q[15];
ry(3.0006654179328205) q[14];
ry(-2.5094317370643178) q[15];
cx q[14],q[15];
ry(-0.1538265322671564) q[16];
ry(-0.9437888214000454) q[17];
cx q[16],q[17];
ry(-0.3038160884173049) q[16];
ry(0.1133992754438025) q[17];
cx q[16],q[17];
ry(2.966255259829244) q[18];
ry(2.931755696071092) q[19];
cx q[18],q[19];
ry(1.5442845859359622) q[18];
ry(-2.015107581778895) q[19];
cx q[18],q[19];
ry(-2.172397863588707) q[1];
ry(2.9470205179105164) q[2];
cx q[1],q[2];
ry(0.0051875372821656995) q[1];
ry(-1.7524811192049805) q[2];
cx q[1],q[2];
ry(2.025976497753205) q[3];
ry(-2.7357867520711485) q[4];
cx q[3],q[4];
ry(6.171482684702665e-05) q[3];
ry(0.0021252549070815974) q[4];
cx q[3],q[4];
ry(1.0159668303339495) q[5];
ry(0.8301238324450365) q[6];
cx q[5],q[6];
ry(-0.022551541649892395) q[5];
ry(0.5777507488682581) q[6];
cx q[5],q[6];
ry(-0.5361564426542624) q[7];
ry(1.582702743347742) q[8];
cx q[7],q[8];
ry(2.4433476786683967) q[7];
ry(-0.01927256892717288) q[8];
cx q[7],q[8];
ry(-2.3056018070345243) q[9];
ry(1.5923229742140141) q[10];
cx q[9],q[10];
ry(2.547240189335351) q[9];
ry(1.55075814057093) q[10];
cx q[9],q[10];
ry(-1.5695592464378356) q[11];
ry(-2.5095286708941593) q[12];
cx q[11],q[12];
ry(3.135372598227891) q[11];
ry(-0.7295562142534147) q[12];
cx q[11],q[12];
ry(-1.5732624887667486) q[13];
ry(1.4986228441348528) q[14];
cx q[13],q[14];
ry(1.5578424637478956) q[13];
ry(-2.6361821358018647) q[14];
cx q[13],q[14];
ry(-0.13731763681433173) q[15];
ry(-2.614773969293197) q[16];
cx q[15],q[16];
ry(-1.3628028990035022) q[15];
ry(3.1096707228753404) q[16];
cx q[15],q[16];
ry(0.8473566093263551) q[17];
ry(-0.0059491438834616195) q[18];
cx q[17],q[18];
ry(-0.3958883691335793) q[17];
ry(3.0363475133972684) q[18];
cx q[17],q[18];
ry(2.854049361279314) q[0];
ry(-1.1744014388553898) q[1];
cx q[0],q[1];
ry(0.529803147510017) q[0];
ry(-2.039364523868525) q[1];
cx q[0],q[1];
ry(1.2721416479875256) q[2];
ry(1.7507846178232451) q[3];
cx q[2],q[3];
ry(1.4058056152096974) q[2];
ry(-2.0712094307964515) q[3];
cx q[2],q[3];
ry(-1.2844695768467669) q[4];
ry(1.5608249275310992) q[5];
cx q[4],q[5];
ry(1.838321611324199) q[4];
ry(3.12648423067742) q[5];
cx q[4],q[5];
ry(2.287593478963975) q[6];
ry(-2.851435565530489) q[7];
cx q[6],q[7];
ry(1.9410476337299107) q[6];
ry(-0.4690919093841936) q[7];
cx q[6],q[7];
ry(1.9500678848355675) q[8];
ry(-2.8598208381967085) q[9];
cx q[8],q[9];
ry(5.7365242517981585e-05) q[8];
ry(-3.1389106811746106) q[9];
cx q[8],q[9];
ry(-1.359865823053215) q[10];
ry(-0.8232918100358609) q[11];
cx q[10],q[11];
ry(-3.135691602820693) q[10];
ry(0.007844842233940207) q[11];
cx q[10],q[11];
ry(1.9833763818319636) q[12];
ry(-1.5742320640763332) q[13];
cx q[12],q[13];
ry(2.668335662170995) q[12];
ry(2.0864604569626413) q[13];
cx q[12],q[13];
ry(0.5865031021601492) q[14];
ry(2.6574726979353054) q[15];
cx q[14],q[15];
ry(-1.7134424017859855) q[14];
ry(3.1405956281321297) q[15];
cx q[14],q[15];
ry(-1.998269589984164) q[16];
ry(0.02670118096805751) q[17];
cx q[16],q[17];
ry(-0.20142303595053615) q[16];
ry(-0.23633363775323302) q[17];
cx q[16],q[17];
ry(-1.062487070322061) q[18];
ry(0.7218641826119433) q[19];
cx q[18],q[19];
ry(3.1255270430945195) q[18];
ry(1.4395314251375906) q[19];
cx q[18],q[19];
ry(1.378688650070524) q[1];
ry(2.7541172807816094) q[2];
cx q[1],q[2];
ry(-0.20047208263958916) q[1];
ry(0.014907485922397434) q[2];
cx q[1],q[2];
ry(-3.0712044712808426) q[3];
ry(-2.582181619436181) q[4];
cx q[3],q[4];
ry(-0.2326063333912165) q[3];
ry(-3.128462109826197) q[4];
cx q[3],q[4];
ry(1.5989872437881303) q[5];
ry(1.51633625347605) q[6];
cx q[5],q[6];
ry(-1.1859379005778314) q[5];
ry(-0.7047407375784616) q[6];
cx q[5],q[6];
ry(1.558062377969086) q[7];
ry(-1.1920362460538039) q[8];
cx q[7],q[8];
ry(-2.254768883225299) q[7];
ry(1.2260416202473658) q[8];
cx q[7],q[8];
ry(1.1735289103794198) q[9];
ry(-1.7547997695066337) q[10];
cx q[9],q[10];
ry(2.2588587131113336) q[9];
ry(-2.5174916172767268) q[10];
cx q[9],q[10];
ry(-0.819258656310167) q[11];
ry(3.0154812769141626) q[12];
cx q[11],q[12];
ry(3.1386947048656886) q[11];
ry(2.0333368470461046) q[12];
cx q[11],q[12];
ry(1.5663414092466805) q[13];
ry(0.5875779317184815) q[14];
cx q[13],q[14];
ry(1.1036286095315029) q[13];
ry(-1.9792233753801654) q[14];
cx q[13],q[14];
ry(1.5716178680739543) q[15];
ry(1.6848139381198655) q[16];
cx q[15],q[16];
ry(1.5474195641655044) q[15];
ry(-1.6167826191024757) q[16];
cx q[15],q[16];
ry(-1.3094353904736495) q[17];
ry(-0.2249845860898351) q[18];
cx q[17],q[18];
ry(-2.2383052702656077) q[17];
ry(1.5536368368486775) q[18];
cx q[17],q[18];
ry(2.9335555087195413) q[0];
ry(-2.2855596302056806) q[1];
cx q[0],q[1];
ry(-2.858675332533019) q[0];
ry(-2.0466784179434647) q[1];
cx q[0],q[1];
ry(-1.6253806960824193) q[2];
ry(-2.9493776174295805) q[3];
cx q[2],q[3];
ry(-3.1309299594254085) q[2];
ry(-1.2614881114147953) q[3];
cx q[2],q[3];
ry(-1.5536140975829922) q[4];
ry(2.255330535960935) q[5];
cx q[4],q[5];
ry(-3.128095099569312) q[4];
ry(1.8062676090608605) q[5];
cx q[4],q[5];
ry(1.5725089718165186) q[6];
ry(-1.5315984728334948) q[7];
cx q[6],q[7];
ry(0.423276451712165) q[6];
ry(0.9146963685204784) q[7];
cx q[6],q[7];
ry(-1.5641334813431058) q[8];
ry(3.0366393594275665) q[9];
cx q[8],q[9];
ry(1.582151549025526) q[8];
ry(-1.3735959286718042) q[9];
cx q[8],q[9];
ry(1.5789978539184077) q[10];
ry(3.1393823975411816) q[11];
cx q[10],q[11];
ry(7.097023587444168e-05) q[10];
ry(2.6177884668250386) q[11];
cx q[10],q[11];
ry(-0.12401373836982632) q[12];
ry(1.5733987100765405) q[13];
cx q[12],q[13];
ry(1.393098934668151) q[12];
ry(2.120493206835022) q[13];
cx q[12],q[13];
ry(-1.5752204045467648) q[14];
ry(-1.56428033936524) q[15];
cx q[14],q[15];
ry(1.3989083336923915) q[14];
ry(-2.583576762612303) q[15];
cx q[14],q[15];
ry(2.6566484964103854) q[16];
ry(1.680387339183128) q[17];
cx q[16],q[17];
ry(-2.230590903013784) q[16];
ry(3.1383170014336277) q[17];
cx q[16],q[17];
ry(1.2884185275591078) q[18];
ry(0.5751223524957733) q[19];
cx q[18],q[19];
ry(-0.6143942884332007) q[18];
ry(-1.2267598841727443) q[19];
cx q[18],q[19];
ry(-2.052081664323392) q[1];
ry(2.964067453488049) q[2];
cx q[1],q[2];
ry(0.38309819293253883) q[1];
ry(0.7996998335988116) q[2];
cx q[1],q[2];
ry(2.4956782680314964) q[3];
ry(-1.7976575109161894) q[4];
cx q[3],q[4];
ry(3.113347812481875) q[3];
ry(-1.5101803173623933) q[4];
cx q[3],q[4];
ry(2.239458219335704) q[5];
ry(-1.5719313541978017) q[6];
cx q[5],q[6];
ry(-0.694298339867558) q[5];
ry(-1.744920333552372) q[6];
cx q[5],q[6];
ry(0.6868086204607707) q[7];
ry(-3.0684220368022728) q[8];
cx q[7],q[8];
ry(2.1279547123322846) q[7];
ry(0.0084724228496178) q[8];
cx q[7],q[8];
ry(1.7811382920780252) q[9];
ry(0.03308803108099017) q[10];
cx q[9],q[10];
ry(0.0016351953951332379) q[9];
ry(-2.264417738091213) q[10];
cx q[9],q[10];
ry(-1.983472140998158) q[11];
ry(-1.57001820622604) q[12];
cx q[11],q[12];
ry(-1.9817497532368549) q[11];
ry(3.1394759124789084) q[12];
cx q[11],q[12];
ry(-1.5717367067078527) q[13];
ry(1.5458940768440907) q[14];
cx q[13],q[14];
ry(0.18026138102483458) q[13];
ry(0.9737924755399928) q[14];
cx q[13],q[14];
ry(2.858678524442514) q[15];
ry(-0.47561468074801466) q[16];
cx q[15],q[16];
ry(2.96996807294789) q[15];
ry(3.1396948489554575) q[16];
cx q[15],q[16];
ry(1.5577888326245048) q[17];
ry(-1.5442960004255184) q[18];
cx q[17],q[18];
ry(-0.017308434660990284) q[17];
ry(2.173749705876613) q[18];
cx q[17],q[18];
ry(-1.6711189453292867) q[0];
ry(1.7795974122217) q[1];
cx q[0],q[1];
ry(-2.8095185753746454) q[0];
ry(0.6963881880023441) q[1];
cx q[0],q[1];
ry(0.6806832547857483) q[2];
ry(-0.7497720104373812) q[3];
cx q[2],q[3];
ry(-3.1386954199330535) q[2];
ry(-3.12180762659105) q[3];
cx q[2],q[3];
ry(-1.8147839562837949) q[4];
ry(1.5733309714684596) q[5];
cx q[4],q[5];
ry(-1.5627643628802383) q[4];
ry(-1.4861472672081124) q[5];
cx q[4],q[5];
ry(-1.562424824098998) q[6];
ry(-2.380550132509758) q[7];
cx q[6],q[7];
ry(-0.08044671255406932) q[6];
ry(2.93360950589349) q[7];
cx q[6],q[7];
ry(-0.8011786088807379) q[8];
ry(-1.2290786428400553) q[9];
cx q[8],q[9];
ry(-0.0025271534866702083) q[8];
ry(0.0015982453369280947) q[9];
cx q[8],q[9];
ry(2.758356673339941) q[10];
ry(-0.4162928111745688) q[11];
cx q[10],q[11];
ry(1.5900777729518145) q[10];
ry(0.009104594598434266) q[11];
cx q[10],q[11];
ry(-1.5610350657997785) q[12];
ry(-1.5705770821136351) q[13];
cx q[12],q[13];
ry(-1.1982204959494132) q[12];
ry(-0.3669028323193983) q[13];
cx q[12],q[13];
ry(-1.547565665035604) q[14];
ry(-0.250097654735205) q[15];
cx q[14],q[15];
ry(-0.24250273058744745) q[14];
ry(-0.980416512015475) q[15];
cx q[14],q[15];
ry(-1.5590533415735721) q[16];
ry(1.4376433716453456) q[17];
cx q[16],q[17];
ry(-0.4407088058293296) q[16];
ry(0.6438787343084691) q[17];
cx q[16],q[17];
ry(1.3205849848894653) q[18];
ry(-2.7623939574978893) q[19];
cx q[18],q[19];
ry(0.9060591655151143) q[18];
ry(2.9370070506019825) q[19];
cx q[18],q[19];
ry(-2.0679664829656055) q[1];
ry(2.985859086522419) q[2];
cx q[1],q[2];
ry(-1.66685329961481) q[1];
ry(1.998201507132723) q[2];
cx q[1],q[2];
ry(2.3956272785974404) q[3];
ry(1.570590889599724) q[4];
cx q[3],q[4];
ry(-1.7666522530466393) q[3];
ry(-1.8966856431049364) q[4];
cx q[3],q[4];
ry(1.5720853660875094) q[5];
ry(-1.5684845300837589) q[6];
cx q[5],q[6];
ry(-2.081669590762182) q[5];
ry(1.8766433463537746) q[6];
cx q[5],q[6];
ry(-3.0640267625317437) q[7];
ry(-0.6978161375582668) q[8];
cx q[7],q[8];
ry(1.6123819057760267) q[7];
ry(1.562861182014788) q[8];
cx q[7],q[8];
ry(2.4867247614377908) q[9];
ry(-1.2284903798885944) q[10];
cx q[9],q[10];
ry(1.5305024610660616) q[9];
ry(-2.9626344366362294) q[10];
cx q[9],q[10];
ry(1.5685605627435333) q[11];
ry(-1.5843423593306616) q[12];
cx q[11],q[12];
ry(1.9510061170076003) q[11];
ry(1.3487333084266346) q[12];
cx q[11],q[12];
ry(1.555685750289432) q[13];
ry(-1.572412396239471) q[14];
cx q[13],q[14];
ry(1.4049360825996597) q[13];
ry(-1.156736764824572) q[14];
cx q[13],q[14];
ry(0.671906947676057) q[15];
ry(-2.187701915495707) q[16];
cx q[15],q[16];
ry(0.015549291255737785) q[15];
ry(0.0038255577665893266) q[16];
cx q[15],q[16];
ry(1.782056726398741) q[17];
ry(-1.7493729939953233) q[18];
cx q[17],q[18];
ry(-0.35709332588608245) q[17];
ry(-1.765367580065525) q[18];
cx q[17],q[18];
ry(0.27619585886517845) q[0];
ry(1.7505874490598514) q[1];
cx q[0],q[1];
ry(-1.5341006753587254) q[0];
ry(2.8060343219413424) q[1];
cx q[0],q[1];
ry(1.4326160019492233) q[2];
ry(-1.5706208804642656) q[3];
cx q[2],q[3];
ry(1.614172646689183) q[2];
ry(-2.3382151344252495) q[3];
cx q[2],q[3];
ry(0.7723184834200882) q[4];
ry(1.568215655599845) q[5];
cx q[4],q[5];
ry(-1.5857162508371794) q[4];
ry(0.00031519331600741193) q[5];
cx q[4],q[5];
ry(1.5708276839138235) q[6];
ry(1.5782345075278348) q[7];
cx q[6],q[7];
ry(-1.432383438385223) q[6];
ry(-1.3730111901749968) q[7];
cx q[6],q[7];
ry(-2.5805390579898724) q[8];
ry(2.7006807660952687) q[9];
cx q[8],q[9];
ry(2.6740404124525536) q[8];
ry(2.3014680280860214) q[9];
cx q[8],q[9];
ry(-1.5694472138440116) q[10];
ry(2.104608960218035) q[11];
cx q[10],q[11];
ry(3.1406339132908987) q[10];
ry(1.953266113120768) q[11];
cx q[10],q[11];
ry(-0.9937094754779403) q[12];
ry(-1.5621485953623189) q[13];
cx q[12],q[13];
ry(0.17244451549547613) q[12];
ry(-3.1373675195498585) q[13];
cx q[12],q[13];
ry(-1.5683465027159702) q[14];
ry(2.4330714073246) q[15];
cx q[14],q[15];
ry(0.6158580467895893) q[14];
ry(-0.12435154518828785) q[15];
cx q[14],q[15];
ry(-0.955887259591326) q[16];
ry(-1.5036261687976564) q[17];
cx q[16],q[17];
ry(1.1056961519807755) q[16];
ry(-2.178768350713267) q[17];
cx q[16],q[17];
ry(-0.19669540000462005) q[18];
ry(1.4102522605835004) q[19];
cx q[18],q[19];
ry(1.3709641537201591) q[18];
ry(-3.0022588921698663) q[19];
cx q[18],q[19];
ry(-1.5295623504196252) q[1];
ry(-1.5823781611003307) q[2];
cx q[1],q[2];
ry(0.9925826017146155) q[1];
ry(-0.7090927696827917) q[2];
cx q[1],q[2];
ry(-2.532244885728148) q[3];
ry(-0.7717277239325409) q[4];
cx q[3],q[4];
ry(1.5723971239674224) q[3];
ry(0.0024879092124612256) q[4];
cx q[3],q[4];
ry(1.5699348691631183) q[5];
ry(-1.571097482250476) q[6];
cx q[5],q[6];
ry(-1.8374878552146692) q[5];
ry(0.5346069314596473) q[6];
cx q[5],q[6];
ry(0.21572448545617817) q[7];
ry(-1.5696872374343025) q[8];
cx q[7],q[8];
ry(-0.6005029163122346) q[7];
ry(-3.1408546674809443) q[8];
cx q[7],q[8];
ry(-1.5752061724198043) q[9];
ry(-1.5704215556078431) q[10];
cx q[9],q[10];
ry(-1.2401066272033505) q[9];
ry(-1.7360500028245762) q[10];
cx q[9],q[10];
ry(2.095389560141667) q[11];
ry(2.182588606246699) q[12];
cx q[11],q[12];
ry(0.1349037568947571) q[11];
ry(2.920267747593852) q[12];
cx q[11],q[12];
ry(-0.8268165572357589) q[13];
ry(2.9013919732445244) q[14];
cx q[13],q[14];
ry(1.9354709130819117) q[13];
ry(0.8871115116188794) q[14];
cx q[13],q[14];
ry(1.5714552253912297) q[15];
ry(-1.5761391129667306) q[16];
cx q[15],q[16];
ry(-1.882216683446006) q[15];
ry(-1.9652483453040162) q[16];
cx q[15],q[16];
ry(1.5737955803398105) q[17];
ry(2.9791684853893217) q[18];
cx q[17],q[18];
ry(1.673791106726953) q[17];
ry(0.7902721186565562) q[18];
cx q[17],q[18];
ry(-1.5309796534028843) q[0];
ry(1.5184157718909441) q[1];
cx q[0],q[1];
ry(-1.336068972352554) q[0];
ry(1.5519867850782632) q[1];
cx q[0],q[1];
ry(-1.5703193944659457) q[2];
ry(2.535469354378962) q[3];
cx q[2],q[3];
ry(2.4162250181997305) q[2];
ry(1.1646346455111738) q[3];
cx q[2],q[3];
ry(1.5720731903729437) q[4];
ry(1.5725267968319445) q[5];
cx q[4],q[5];
ry(2.5624280960011894) q[4];
ry(-1.5378400527694813) q[5];
cx q[4],q[5];
ry(2.1752787933380677) q[6];
ry(0.2157414715268473) q[7];
cx q[6],q[7];
ry(-1.6260173774555486) q[6];
ry(0.000722069557917101) q[7];
cx q[6],q[7];
ry(2.3907589061443266) q[8];
ry(1.5669794826483043) q[9];
cx q[8],q[9];
ry(0.41955886967731865) q[8];
ry(-3.1415332609758577) q[9];
cx q[8],q[9];
ry(1.5794124097807025) q[10];
ry(1.5626779828177604) q[11];
cx q[10],q[11];
ry(-1.6009546123209253) q[10];
ry(2.8642900661000437) q[11];
cx q[10],q[11];
ry(-1.532953989999192) q[12];
ry(2.7770643941464854) q[13];
cx q[12],q[13];
ry(-0.0014306287608443216) q[12];
ry(0.005036152074112009) q[13];
cx q[12],q[13];
ry(-2.3774030471312977) q[14];
ry(2.991173396404851) q[15];
cx q[14],q[15];
ry(3.1397695252981688) q[14];
ry(0.015291175607142549) q[15];
cx q[14],q[15];
ry(1.5669624873686971) q[16];
ry(-1.6049353550029846) q[17];
cx q[16],q[17];
ry(3.046430496479913) q[16];
ry(-2.42702113998975) q[17];
cx q[16],q[17];
ry(-1.4803267946152614) q[18];
ry(2.26398604068151) q[19];
cx q[18],q[19];
ry(1.5699686514805105) q[18];
ry(3.052414034622671) q[19];
cx q[18],q[19];
ry(3.127686367654367) q[1];
ry(1.627754458448031) q[2];
cx q[1],q[2];
ry(-0.013033971164657638) q[1];
ry(-0.0014670571460541635) q[2];
cx q[1],q[2];
ry(1.5730497338459744) q[3];
ry(-1.5712114858927606) q[4];
cx q[3],q[4];
ry(-1.5768975819256645) q[3];
ry(-1.6077504437038899) q[4];
cx q[3],q[4];
ry(-1.570121603483301) q[5];
ry(-2.16260068964444) q[6];
cx q[5],q[6];
ry(-0.10571762836611075) q[5];
ry(-1.5998519336051595) q[6];
cx q[5],q[6];
ry(1.544402675899783) q[7];
ry(0.016637860480791744) q[8];
cx q[7],q[8];
ry(-3.141269181939313) q[7];
ry(3.111319472916689) q[8];
cx q[7],q[8];
ry(1.5578304773372462) q[9];
ry(-1.563601311920084) q[10];
cx q[9],q[10];
ry(1.5948765533450608) q[9];
ry(2.7955709036256486) q[10];
cx q[9],q[10];
ry(0.7274627942190497) q[11];
ry(1.498387931365841) q[12];
cx q[11],q[12];
ry(-1.522350893559211) q[11];
ry(-0.0007960125079824465) q[12];
cx q[11],q[12];
ry(3.00824205294054) q[13];
ry(2.6656836553682304) q[14];
cx q[13],q[14];
ry(-1.2876687599957695) q[13];
ry(1.5989858181099388) q[14];
cx q[13],q[14];
ry(-0.14901790527028955) q[15];
ry(1.580263906081333) q[16];
cx q[15],q[16];
ry(-0.19352302507536945) q[15];
ry(1.0081444397265116) q[16];
cx q[15],q[16];
ry(1.4031796100815903) q[17];
ry(-1.4760134421425137) q[18];
cx q[17],q[18];
ry(-2.724893600440881) q[17];
ry(3.08407459820073) q[18];
cx q[17],q[18];
ry(2.922581731244792) q[0];
ry(-3.1324321746745767) q[1];
ry(3.082519615358426) q[2];
ry(1.5714101246283505) q[3];
ry(-3.1403416336467607) q[4];
ry(1.570872842882592) q[5];
ry(3.1280861313936748) q[6];
ry(1.544834325199117) q[7];
ry(-2.375398489778747) q[8];
ry(-1.5873683171202257) q[9];
ry(-3.1408299131779245) q[10];
ry(2.4148844250509724) q[11];
ry(-3.1409389533942864) q[12];
ry(-1.0055893698908418) q[13];
ry(1.2386398002174965) q[14];
ry(-1.569575140718979) q[15];
ry(-0.013563345002448192) q[16];
ry(-1.7176474454815263) q[17];
ry(-3.1390725549791996) q[18];
ry(1.6882925738819563) q[19];