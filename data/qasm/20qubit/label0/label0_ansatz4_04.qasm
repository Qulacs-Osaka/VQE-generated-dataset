OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.079455569871453) q[0];
rz(1.952760649189602) q[0];
ry(-0.001804370236889952) q[1];
rz(-1.0203951338622528) q[1];
ry(-0.35150308066765845) q[2];
rz(-3.1386730041816158) q[2];
ry(0.8463116510040419) q[3];
rz(-3.140194744919571) q[3];
ry(2.4467481663847938e-05) q[4];
rz(-2.4578803600274997) q[4];
ry(-8.913220419714506e-07) q[5];
rz(-1.3122414539160294) q[5];
ry(1.5717760314875617) q[6];
rz(-0.00493196308692312) q[6];
ry(0.2404518721653122) q[7];
rz(0.01833656861527286) q[7];
ry(1.5710039764153834) q[8];
rz(1.570829081400615) q[8];
ry(1.570298559573506) q[9];
rz(-1.5707931836124567) q[9];
ry(1.4639673335815526) q[10];
rz(-1.6868197965504201) q[10];
ry(-1.5700325715937753) q[11];
rz(1.694528228390914) q[11];
ry(1.5707663233648042) q[12];
rz(-1.570182713328717) q[12];
ry(0.24987134595847899) q[13];
rz(1.152499964337646) q[13];
ry(-3.1415096133618423) q[14];
rz(2.2940572219721065) q[14];
ry(-2.4303677602866003e-05) q[15];
rz(-0.042649622429705394) q[15];
ry(-3.1408603669286683) q[16];
rz(0.5865556927626135) q[16];
ry(-0.29637155086306066) q[17];
rz(-0.7620156348796608) q[17];
ry(-0.45310221399637746) q[18];
rz(-2.073406409334771) q[18];
ry(-2.9295436389406597) q[19];
rz(0.7943225503991371) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5736073297493665) q[0];
rz(1.3067651787683117) q[0];
ry(-0.2731002717995054) q[1];
rz(-0.44794046337262833) q[1];
ry(0.239135038153328) q[2];
rz(-1.5187816577240285) q[2];
ry(-0.8121647958950231) q[3];
rz(-1.6062426129819962) q[3];
ry(1.147488713413269) q[4];
rz(1.063565729598842) q[4];
ry(-0.0028186641247245348) q[5];
rz(-1.9103232928632696) q[5];
ry(0.28250341361280695) q[6];
rz(-0.0019057778441376161) q[6];
ry(-0.10375555706506034) q[7];
rz(-2.1541527823891626) q[7];
ry(1.5707230221185007) q[8];
rz(-2.1254288688398164) q[8];
ry(1.5708378264102922) q[9];
rz(-2.125424956174468) q[9];
ry(-3.1415628413128327) q[10];
rz(3.0254615608961957) q[10];
ry(1.402992155163929e-05) q[11];
rz(1.7946548601490073) q[11];
ry(3.088547268957053) q[12];
rz(0.0005441633780085394) q[12];
ry(-1.257457490939675e-06) q[13];
rz(1.893618434579099) q[13];
ry(-2.3132593432474757) q[14];
rz(-3.0120282188678886) q[14];
ry(-0.35543782132981416) q[15];
rz(2.560154520955732) q[15];
ry(3.141213524765013) q[16];
rz(2.4835014341253836) q[16];
ry(-0.3083482373325239) q[17];
rz(-3.015755250673878) q[17];
ry(-1.7712019453399455) q[18];
rz(-2.9154328480530194) q[18];
ry(3.0796628307309817) q[19];
rz(-2.236396773410571) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.089503825325377) q[0];
rz(0.38699583434063284) q[0];
ry(-1.321848732950051) q[1];
rz(-1.5624426350021317) q[1];
ry(3.112648477379917) q[2];
rz(0.17741581238332424) q[2];
ry(-2.9065233544305746) q[3];
rz(0.10801056151458886) q[3];
ry(0.00013634411772175397) q[4];
rz(2.033420426173425) q[4];
ry(3.141586521910527) q[5];
rz(1.6993369173150505) q[5];
ry(0.024448529471479574) q[6];
rz(0.007068824175959128) q[6];
ry(3.141560441923714) q[7];
rz(1.2130881025675797) q[7];
ry(1.5703613865910153) q[8];
rz(3.120646520657887) q[8];
ry(1.570608137673056) q[9];
rz(-3.052842115431863) q[9];
ry(1.7118136271914661) q[10];
rz(-1.0756608708042148) q[10];
ry(2.813699792425854) q[11];
rz(-1.328945963687012) q[11];
ry(1.3618579196464973) q[12];
rz(1.5707612760751912) q[12];
ry(0.00111795718324992) q[13];
rz(-3.0414705758882503) q[13];
ry(1.4787764357926393e-05) q[14];
rz(-1.8976843663898315) q[14];
ry(3.1415548887473586) q[15];
rz(-0.21129355999770627) q[15];
ry(-7.183984184624986e-05) q[16];
rz(-1.396536748603162) q[16];
ry(3.137373362307381) q[17];
rz(3.098472017047627) q[17];
ry(1.771308809830308) q[18];
rz(-0.967955022993693) q[18];
ry(0.028422749245837764) q[19];
rz(-1.227357532746753) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.3337721207021866) q[0];
rz(-1.4688327247875481) q[0];
ry(2.166756923225715) q[1];
rz(1.4751268848684846) q[1];
ry(3.1362434906194983) q[2];
rz(0.6290116252723283) q[2];
ry(-3.129990358840432) q[3];
rz(3.1208372936527087) q[3];
ry(1.6050034812248875) q[4];
rz(3.0444752986922454) q[4];
ry(3.0689229541886336) q[5];
rz(0.037069535240177316) q[5];
ry(1.5718477066435232) q[6];
rz(-2.4714516046169233) q[6];
ry(-0.0002583925073444959) q[7];
rz(2.9483578622099533) q[7];
ry(-0.00869879078897463) q[8];
rz(0.020881581183284773) q[8];
ry(-0.0036424036277171723) q[9];
rz(1.4941726539021893) q[9];
ry(-3.1415529451573008) q[10];
rz(-2.134213820051963) q[10];
ry(-3.1415552041825547) q[11];
rz(1.259442141111659) q[11];
ry(-1.5707808562677696) q[12];
rz(1.6317281880753312) q[12];
ry(1.5707369580575588) q[13];
rz(-1.5707859856272464) q[13];
ry(-0.7705725705601649) q[14];
rz(-1.235634815381247) q[14];
ry(-0.05487099901029335) q[15];
rz(-0.012875389483428801) q[15];
ry(-1.5665369510152705) q[16];
rz(-1.5749106761057003) q[16];
ry(2.9991630009143337) q[17];
rz(-2.501137528080669) q[17];
ry(-2.800000305976961) q[18];
rz(2.3946752365097805) q[18];
ry(0.02916731038528031) q[19];
rz(0.5363942736313065) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.8916773704519436) q[0];
rz(-2.6686832123303734) q[0];
ry(1.3764275346723371) q[1];
rz(-0.1691359426298155) q[1];
ry(0.00023385595181046884) q[2];
rz(1.1018943457038084) q[2];
ry(-0.0005871264773839907) q[3];
rz(-1.4615354026572547) q[3];
ry(-0.2055375665750223) q[4];
rz(-3.0999661148065107) q[4];
ry(1.5806558979642644) q[5];
rz(-0.028516204984752814) q[5];
ry(0.0028982227214237174) q[6];
rz(-2.276079894405691) q[6];
ry(0.001034742534932853) q[7];
rz(1.540657477825576) q[7];
ry(1.5706790735538294) q[8];
rz(1.803909601165736) q[8];
ry(-0.02974848419916576) q[9];
rz(3.1294186597732465) q[9];
ry(0.0004949971077943261) q[10];
rz(-0.512147886132604) q[10];
ry(0.012852893934083783) q[11];
rz(0.31211573724878333) q[11];
ry(-3.1410907440435163) q[12];
rz(1.6383918860616848) q[12];
ry(-1.3087979206016556) q[13];
rz(1.5708514487475185) q[13];
ry(-0.21016105963889603) q[14];
rz(0.47826195031438506) q[14];
ry(-0.02421386017866123) q[15];
rz(1.0723609386741035) q[15];
ry(2.814661805298249) q[16];
rz(-1.4423661357413655) q[16];
ry(-3.140868030869539) q[17];
rz(-0.5306299465432529) q[17];
ry(-3.1412231126326096) q[18];
rz(0.019643625010905268) q[18];
ry(-0.0001714495756957035) q[19];
rz(-1.3470623965543531) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.2452021527229977) q[0];
rz(2.3067486085102673) q[0];
ry(-1.7012555478070848) q[1];
rz(0.7819058262290642) q[1];
ry(0.44570497840447404) q[2];
rz(-2.2088688977915614) q[2];
ry(3.0885237255506457) q[3];
rz(-1.2838161810545103) q[3];
ry(1.5019570515288416) q[4];
rz(-0.3682291422471826) q[4];
ry(1.7472227573385544) q[5];
rz(2.176210286713024) q[5];
ry(0.027697493072344592) q[6];
rz(-0.6254863468359755) q[6];
ry(2.913119609097526) q[7];
rz(3.1114266221511815) q[7];
ry(1.56973300445856) q[8];
rz(0.006171103681454824) q[8];
ry(1.5880304983909361) q[9];
rz(0.00011695447577597663) q[9];
ry(-2.404504730973222) q[10];
rz(-3.1414417398640366) q[10];
ry(-0.7675168005591153) q[11];
rz(-0.08723156173852993) q[11];
ry(1.570803431360642) q[12];
rz(1.5168444958199763) q[12];
ry(-1.5708228343040085) q[13];
rz(0.9474738560200313) q[13];
ry(-0.016584068627188774) q[14];
rz(-1.992613268180861) q[14];
ry(0.007628880959052785) q[15];
rz(-0.0380049510010816) q[15];
ry(-2.896160879117815e-06) q[16];
rz(1.436045492825751) q[16];
ry(-1.5729035524393824) q[17];
rz(-1.4364770470094075) q[17];
ry(-3.0526404452578713) q[18];
rz(-1.3353429185658083) q[18];
ry(-3.11867571592178) q[19];
rz(-3.031554633276648) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6057641742838644) q[0];
rz(1.525711463561063) q[0];
ry(1.570565441468637) q[1];
rz(-2.1007975090965934) q[1];
ry(0.0010265878637722992) q[2];
rz(-2.5008510403032274) q[2];
ry(-0.005291987067939064) q[3];
rz(2.8413545997660288) q[3];
ry(3.141192762886742) q[4];
rz(0.41596897247135095) q[4];
ry(6.0834324900227437e-05) q[5];
rz(0.97729918266496) q[5];
ry(-6.58081576325742e-05) q[6];
rz(2.214203666623198) q[6];
ry(2.1662166293435433e-05) q[7];
rz(-1.5377481454606459) q[7];
ry(1.5707273942342441) q[8];
rz(0.22056439975386707) q[8];
ry(1.5717057697032175) q[9];
rz(1.5708934422503469) q[9];
ry(1.5821738645381822) q[10];
rz(-3.141289011061345) q[10];
ry(-1.5712861976281856) q[11];
rz(-1.5710972567532508) q[11];
ry(0.0024941776350240643) q[12];
rz(0.05389637393631119) q[12];
ry(3.1414677231634904) q[13];
rz(2.5116003616495313) q[13];
ry(0.002657835089082461) q[14];
rz(1.5225316725893085) q[14];
ry(0.0008532087655536078) q[15];
rz(2.078313712704726) q[15];
ry(3.015765681444369) q[16];
rz(-1.5723212864415301) q[16];
ry(1.5706428403410264) q[17];
rz(-3.1410526251749) q[17];
ry(3.1415570904483685) q[18];
rz(-0.28460483067077297) q[18];
ry(-3.244084199940114e-05) q[19];
rz(0.49278781798267346) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1375682593414087) q[0];
rz(1.471671948402081) q[0];
ry(-0.0025223360498234423) q[1];
rz(-0.48537728285032783) q[1];
ry(-1.5706942556208243) q[2];
rz(-0.370744604621831) q[2];
ry(-1.568321008057391) q[3];
rz(-3.0983004311123605) q[3];
ry(-3.0641226784131605) q[4];
rz(-2.3603279982579957) q[4];
ry(-1.6671594951479296) q[5];
rz(0.027939792945057107) q[5];
ry(1.5717513934483487) q[6];
rz(3.1413231553590317) q[6];
ry(1.570408064646026) q[7];
rz(-3.1415877796559677) q[7];
ry(-0.002101666440275096) q[8];
rz(1.3498568640577888) q[8];
ry(1.5737412479475388) q[9];
rz(1.5707727537073095) q[9];
ry(-1.5707518811819625) q[10];
rz(-3.1415649764151) q[10];
ry(1.5709083047889656) q[11];
rz(-2.6559229196240977e-05) q[11];
ry(-1.5709186621565385) q[12];
rz(1.5706603861838824) q[12];
ry(-1.5707941729739128) q[13];
rz(-4.796868107916064e-05) q[13];
ry(1.5707662639884186) q[14];
rz(-3.141492774862625) q[14];
ry(1.5708846930964744) q[15];
rz(-0.001042846913947848) q[15];
ry(1.8310463698654171) q[16];
rz(-1.5737551053113554) q[16];
ry(-1.6757699578540235) q[17];
rz(1.5699756566171423) q[17];
ry(-1.5708421885448132) q[18];
rz(-3.1414669497868117) q[18];
ry(1.5708002757566637) q[19];
rz(-4.6949040765437704e-05) q[19];