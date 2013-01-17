
var opts = {};

opts.A = 0;
opts.B = 1;
opts.C = 2;
opts.D = 3;
opts.E = 4;
opts.F = 5;
opts.G = 6;
opts.H = 7;
opts.I = 8;

opts.fA = function( args ) { this.args = args; this.a = args.a; this.type = opts.A; }; opts.fA.prototype = { constructor: opts.fA, op: function( context ) { context.add( this.a ); } };
opts.fB = function( args ) { this.args = args; this.b = args.b; this.type = opts.B; }; opts.fB.prototype = { constructor: opts.fB, op: function( context ) { context.add( this.b ); } };
opts.fC = function( args ) { this.args = args; this.c = args.c; this.type = opts.C; }; opts.fC.prototype = { constructor: opts.fC, op: function( context ) { context.add( this.c ); } };
opts.fD = function( args ) { this.args = args; this.d = args.d; this.type = opts.D; }; opts.fD.prototype = { constructor: opts.fD, op: function( context ) { context.add( this.d ); } };
opts.fE = function( args ) { this.args = args; this.e = args.e; this.type = opts.E; }; opts.fE.prototype = { constructor: opts.fE, op: function( context ) { context.add( this.e ); } };
opts.fF = function( args ) { this.args = args; this.f = args.f; this.type = opts.F; }; opts.fF.prototype = { constructor: opts.fF, op: function( context ) { context.add( this.f ); } };
opts.fG = function( args ) { this.args = args; this.g = args.g; this.type = opts.G; }; opts.fG.prototype = { constructor: opts.fG, op: function( context ) { context.add( this.g ); } };
opts.fH = function( args ) { this.args = args; this.h = args.h; this.type = opts.H; }; opts.fH.prototype = { constructor: opts.fH, op: function( context ) { context.add( this.h ); } };
opts.fI = function( args ) { this.args = args; this.i = args.i; this.type = opts.I; }; opts.fI.prototype = { constructor: opts.fI, op: function( context ) { context.add( this.i ); } };

var Context = function() {
    this.n = 0;
};

Context.prototype = {
    constructor: Context,
    
    add: function( x ) {
        this.n += x;
    }
};

function switchyGeneral( ctx, x ) {
    switch( x.type ) {
        case opts.A: ctx.add( x.args.a ); break;
        case opts.B: ctx.add( x.args.b ); break;
        case opts.C: ctx.add( x.args.c ); break;
        case opts.D: ctx.add( x.args.d ); break;
        case opts.E: ctx.add( x.args.e ); break;
        case opts.F: ctx.add( x.args.f ); break;
        case opts.G: ctx.add( x.args.g ); break;
        case opts.H: ctx.add( x.args.h ); break;
        case opts.I: ctx.add( x.args.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

function switchySpecific( ctx, x ) {
    switch( x.type ) {
        case opts.A: ctx.add( x.a ); break;
        case opts.B: ctx.add( x.b ); break;
        case opts.C: ctx.add( x.c ); break;
        case opts.D: ctx.add( x.d ); break;
        case opts.E: ctx.add( x.e ); break;
        case opts.F: ctx.add( x.f ); break;
        case opts.G: ctx.add( x.g ); break;
        case opts.H: ctx.add( x.h ); break;
        case opts.I: ctx.add( x.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var listOfThings = [];
for( var i = 0; i < 10; i++ ) {
    listOfThings.push( new opts.fA( 20 ) );
    listOfThings.push( new opts.fB( 40 ) );
    listOfThings.push( new opts.fC( 60 ) );
    listOfThings.push( new opts.fD( 80 ) );
    listOfThings.push( new opts.fE( 10 ) );
    listOfThings.push( new opts.fF( 30 ) );
    listOfThings.push( new opts.fG( 50 ) );
    listOfThings.push( new opts.fH( 70 ) );
    listOfThings.push( new opts.fI( 90 ) );
}
var count = listOfThings.length;


var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    var x = listOfThings[i];
    switch( x.type ) {
        case opts.A: ctx.add( x.args.a ); break;
        case opts.B: ctx.add( x.args.b ); break;
        case opts.C: ctx.add( x.args.c ); break;
        case opts.D: ctx.add( x.args.d ); break;
        case opts.E: ctx.add( x.args.e ); break;
        case opts.F: ctx.add( x.args.f ); break;
        case opts.G: ctx.add( x.args.g ); break;
        case opts.H: ctx.add( x.args.h ); break;
        case opts.I: ctx.add( x.args.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    var x = listOfThings[i];
    switch( x.type ) {
        case opts.A: ctx.add( x.a ); break;
        case opts.B: ctx.add( x.b ); break;
        case opts.C: ctx.add( x.c ); break;
        case opts.D: ctx.add( x.d ); break;
        case opts.E: ctx.add( x.e ); break;
        case opts.F: ctx.add( x.f ); break;
        case opts.G: ctx.add( x.g ); break;
        case opts.H: ctx.add( x.h ); break;
        case opts.I: ctx.add( x.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    listOfThings[i].op( ctx );
}


