OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.8424064185175144) q[0];
rz(1.8679960900566241) q[0];
ry(-2.2525239038431244) q[1];
rz(-1.7879051080176023) q[1];
ry(-1.4408513572518062) q[2];
rz(-2.256598055871388) q[2];
ry(1.072269538635541) q[3];
rz(1.2031802107256981) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.05373437342545717) q[0];
rz(-3.095086522707542) q[0];
ry(3.0117367697421273) q[1];
rz(-1.7314833823687605) q[1];
ry(0.16537895061586494) q[2];
rz(1.0516750173665992) q[2];
ry(1.403037982534074) q[3];
rz(2.8062933474596363) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.219502803055025) q[0];
rz(2.693564356434424) q[0];
ry(-1.8532442129168185) q[1];
rz(-2.706935864170023) q[1];
ry(1.9062977019270388) q[2];
rz(2.8904437460776755) q[2];
ry(0.9392719259383213) q[3];
rz(0.54468655057084) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.805331004194918) q[0];
rz(-0.4452740173051542) q[0];
ry(-2.2959766316421444) q[1];
rz(-2.5351354663451726) q[1];
ry(2.0045778500965743) q[2];
rz(1.786407688122981) q[2];
ry(0.021701269869378714) q[3];
rz(3.0880824845375527) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.20153414216653065) q[0];
rz(-1.9142328721926871) q[0];
ry(-2.4795638730456235) q[1];
rz(-2.690753608338585) q[1];
ry(-2.6193993042858685) q[2];
rz(-0.5149800130880892) q[2];
ry(3.043063364547118) q[3];
rz(1.5564398329452818) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.22010993306561213) q[0];
rz(2.825179284749313) q[0];
ry(1.5112302184020612) q[1];
rz(-2.322645960634982) q[1];
ry(-1.7558717213601298) q[2];
rz(-2.3366888102829106) q[2];
ry(0.42331651190517755) q[3];
rz(1.081770277866953) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.1464692769474985) q[0];
rz(-1.2123418914844066) q[0];
ry(0.974774314932177) q[1];
rz(0.8490072474376404) q[1];
ry(-0.6671427606274447) q[2];
rz(-2.38399252389378) q[2];
ry(1.8161523070741297) q[3];
rz(2.4658454491578756) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.70011819071762) q[0];
rz(0.8707828483678847) q[0];
ry(-0.15632664523922646) q[1];
rz(2.2489633986247712) q[1];
ry(0.10262990078929764) q[2];
rz(-1.925436282656361) q[2];
ry(-1.775356546449049) q[3];
rz(2.266674715322605) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.578743624611488) q[0];
rz(-3.0689772682012157) q[0];
ry(2.6203657219436667) q[1];
rz(2.9433724464856423) q[1];
ry(-0.256074662071514) q[2];
rz(-0.8150478346576193) q[2];
ry(1.0382439293086296) q[3];
rz(1.8578081221126832) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3257529420193583) q[0];
rz(1.0976093262674587) q[0];
ry(-1.2233076530889717) q[1];
rz(-1.4223394450853002) q[1];
ry(1.0479202710577018) q[2];
rz(0.944106785207073) q[2];
ry(2.6440674260059636) q[3];
rz(-1.936738519193175) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.092324660782447) q[0];
rz(-2.961458827736715) q[0];
ry(0.960307752514528) q[1];
rz(-2.4722505558644743) q[1];
ry(-1.6302244366182945) q[2];
rz(-0.39339049279994764) q[2];
ry(-0.9876279845281222) q[3];
rz(1.0301560061279) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.4058112185038913) q[0];
rz(-1.3653787855995543) q[0];
ry(-0.897363147500733) q[1];
rz(2.0884527846297987) q[1];
ry(-1.0850890748965396) q[2];
rz(-1.6593374555446054) q[2];
ry(0.08774930298225848) q[3];
rz(0.955440394889493) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.756280481437001) q[0];
rz(0.061556329012233846) q[0];
ry(3.1402036035981147) q[1];
rz(1.7139319967949096) q[1];
ry(-0.4119645200093067) q[2];
rz(2.00418650559812) q[2];
ry(-2.292323940838269) q[3];
rz(-1.607208305963787) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.8115928222446946) q[0];
rz(2.748868140285274) q[0];
ry(-2.6724488182796686) q[1];
rz(-1.6586061643998988) q[1];
ry(-2.203748923624228) q[2];
rz(1.023883538039108) q[2];
ry(1.895382742237565) q[3];
rz(2.102540126398815) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.407750613440595) q[0];
rz(0.2041668041389837) q[0];
ry(1.073971841736825) q[1];
rz(-2.513337907352041) q[1];
ry(-1.9875531712621077) q[2];
rz(-2.620579775690638) q[2];
ry(2.950926882705022) q[3];
rz(-2.7523505572291977) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.5943936053349068) q[0];
rz(-0.28144004206303486) q[0];
ry(2.5770608415072505) q[1];
rz(0.8690033496743037) q[1];
ry(1.606409727630119) q[2];
rz(1.860361248235943) q[2];
ry(-0.932002915257496) q[3];
rz(-1.538104682104239) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3503091269623664) q[0];
rz(-3.1022671930461754) q[0];
ry(2.3021397288860777) q[1];
rz(-1.289956866711703) q[1];
ry(1.4640064109340267) q[2];
rz(-1.3819861711867885) q[2];
ry(-0.6225211738299761) q[3];
rz(0.1123793180418393) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.30477338496007) q[0];
rz(-1.4283785967924914) q[0];
ry(-2.838880615676668) q[1];
rz(-1.0288456941899582) q[1];
ry(-2.420564564169512) q[2];
rz(-0.03705357696943513) q[2];
ry(-1.0239345761551841) q[3];
rz(2.489695160561804) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.49422743073591674) q[0];
rz(-0.47425340228514745) q[0];
ry(1.9388653969588439) q[1];
rz(-0.663495737785732) q[1];
ry(2.5080749968486487) q[2];
rz(-0.0363458418041069) q[2];
ry(-2.4504903565312315) q[3];
rz(0.05039322726405374) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.119382055889554) q[0];
rz(-0.1979816630248008) q[0];
ry(-2.85325368874239) q[1];
rz(3.0236940327019357) q[1];
ry(1.8399580554145611) q[2];
rz(2.10012398725128) q[2];
ry(-0.8645419911394532) q[3];
rz(1.321400388803431) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8565218421818317) q[0];
rz(-0.5048195694626418) q[0];
ry(-0.986192136869611) q[1];
rz(-2.647256476222679) q[1];
ry(0.12865343346914582) q[2];
rz(2.7779643534103307) q[2];
ry(2.9004420477967274) q[3];
rz(2.2766310040739177) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.569945118812885) q[0];
rz(2.34255459065104) q[0];
ry(1.3625558619739067) q[1];
rz(-0.9500127372474534) q[1];
ry(1.5509490539685393) q[2];
rz(-1.3800457752395752) q[2];
ry(1.1586566086070662) q[3];
rz(0.16739851700064268) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.400541953221503) q[0];
rz(1.3402180600446814) q[0];
ry(3.008721189420061) q[1];
rz(0.5523636708901343) q[1];
ry(-1.6212337618365467) q[2];
rz(-1.6354273894999478) q[2];
ry(-2.6327205251626067) q[3];
rz(2.6714302968954775) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.7841741939461517) q[0];
rz(0.44894995773581686) q[0];
ry(-0.49856938999969014) q[1];
rz(1.2534762334208063) q[1];
ry(0.4795153286071043) q[2];
rz(-1.7173118335679005) q[2];
ry(1.7256367135868373) q[3];
rz(1.8713705264298508) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.5585622749132755) q[0];
rz(-0.9891039940037248) q[0];
ry(0.6882604965145971) q[1];
rz(-0.2654416175330896) q[1];
ry(1.0184724057804238) q[2];
rz(-2.7296884528543863) q[2];
ry(-2.7018274452800153) q[3];
rz(-1.2071289267417633) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.7536213252082131) q[0];
rz(-1.0651260773638285) q[0];
ry(0.8471916546558642) q[1];
rz(2.78504983634794) q[1];
ry(1.8035498133637795) q[2];
rz(-1.9389675999002354) q[2];
ry(-1.7124195372211046) q[3];
rz(-3.0587969292258586) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.7678259895677222) q[0];
rz(0.6185585509479115) q[0];
ry(2.971974635801751) q[1];
rz(0.016890343997057574) q[1];
ry(0.8157487155167216) q[2];
rz(-3.021857343349331) q[2];
ry(-0.378617038693333) q[3];
rz(1.6335179347194917) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.9767134584700071) q[0];
rz(2.5424471469476337) q[0];
ry(0.9362453791074943) q[1];
rz(-2.269100312789699) q[1];
ry(0.8378343824018323) q[2];
rz(1.0876162021566478) q[2];
ry(2.4646912122719677) q[3];
rz(-1.1664734751825545) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.011281555656769) q[0];
rz(0.8950246373505505) q[0];
ry(-3.01936655481197) q[1];
rz(0.6829461020819769) q[1];
ry(-2.470208646071288) q[2];
rz(0.1657780576170742) q[2];
ry(0.9527861438380076) q[3];
rz(-1.4983351134639273) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5487332822706161) q[0];
rz(-0.9665674728195659) q[0];
ry(1.3886802900340465) q[1];
rz(-0.5710324915766175) q[1];
ry(-2.04403324369654) q[2];
rz(-2.742294651993008) q[2];
ry(-0.31489631305662247) q[3];
rz(0.5577556965465906) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.120082780682189) q[0];
rz(-2.381836599507408) q[0];
ry(-1.5899012636796108) q[1];
rz(1.1884966358236424) q[1];
ry(0.6070713260189881) q[2];
rz(-0.8687347329569013) q[2];
ry(-1.8701010987083349) q[3];
rz(-0.06117145800539791) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.0212991525716837) q[0];
rz(-2.8754040283206144) q[0];
ry(0.09757234544771676) q[1];
rz(0.6011429319511346) q[1];
ry(0.7275274912850502) q[2];
rz(-1.777347201661485) q[2];
ry(-2.5267735556777198) q[3];
rz(-2.3837939889644404) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1656357266278636) q[0];
rz(1.7763671823492495) q[0];
ry(1.332401903483123) q[1];
rz(0.9791830317670015) q[1];
ry(2.0208798022833383) q[2];
rz(0.08566954566541708) q[2];
ry(-2.4693074248484748) q[3];
rz(0.18468512130647863) q[3];