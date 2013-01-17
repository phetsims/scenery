
var base = {};

base.A = function( str ) {
    this.str = str;
    this.isA = true;
};

base.B = function( n ) {
    this.n = n;
    this.isB = true;
};

base.C = function( a, b ) {
    this.a = a;
    this.b = b;
    this.isC = true;
};

var count = 500;
var aInstances = [];
var bInstances = [];
var cInstances = [];

for( var i = 0; i < count; i++ ) {
    aInstances.push( new base.A( Math.random().toString() ) );
    bInstances.push( new base.B( Math.random() ) );
    cInstances.push( new base.C( Math.random(), Math.random() ) );
}

function fInstanceOf( ob ) {
    if( ob instanceof base.A ) {
        return ob.str.length;
    } else if( ob instanceof base.B ) {
        return ob.n;
    } else if( ob instanceof base.C ) {
        return ob.a * ob.b;
    } else {
        throw new Error( 'unhandled case' );
    }
}

function fChecks( ob ) {
    if( ob.isA ) {
        return ob.str.length;
    } else if( ob.isB ) {
        return ob.n;
    } else if( ob.isC ) {
        return ob.a * ob.b;
    } else {
        throw new Error( 'unhandled case' );
    }
}

function fDuckTyping( ob ) {
    if( ob.str !== undefined ) {
        return ob.str.length;
    } else if( ob.n !== undefined ) {
        return ob.n;
    } else if( ob.a !== undefined && ob.b !== undefined ) {
        return ob.a * ob.b;
    } else {
        throw new Error( 'unhandled case' );
    }
}

function fA( ob ) {
    return ob.str.length;
}
function fB( ob ) {
    return ob.n;
}
function fC( ob ) {
    return ob.a * ob.b;
}



for( var i = 0; i < count; i++ ) {
    fInstanceOf( aInstances[i] );
    fInstanceOf( bInstances[i] );
    fInstanceOf( cInstances[i] );
}

for( var i = 0; i < count; i++ ) {
    fChecks( aInstances[i] );
    fChecks( bInstances[i] );
    fChecks( cInstances[i] );
}

for( var i = 0; i < count; i++ ) {
    fDuckTyping( aInstances[i] );
    fDuckTyping( bInstances[i] );
    fDuckTyping( cInstances[i] );
}

for( var i = 0; i < count; i++ ) {
    fA( aInstances[i] );
    fB( bInstances[i] );
    fC( cInstances[i] );
}




var a = new base.A( 'test string' );
var b = new base.B( 5 );
var c = new base.C( 3, 6 );