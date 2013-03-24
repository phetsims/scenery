// Copyright 2002-2012, University of Colorado

/**
 * An interval between two Trails.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  scenery.TrailInterval = function( a, b ) {
    assert && assert( a.compare( b ) <= 0, 'TrailInterval parameters must not be out of order' );
    
    this.a = a;
    this.b = b;
  };
  var TrailInterval = scenery.TrailInterval;
  
  TrailInterval.prototype = {
    constructor: TrailInterval,
    
    reindex: function() {
      this.a && this.a.reindex();
      this.b && this.b.reindex();
    },
    
    /*
     * Whether the union of this and the specified interval doesn't include any additional trails, when
     * both are treated as exclusive endpoints (exclusive between a and b). We also make the assumption
     * that a !== b || a === null for either interval, since otherwise it is not well defined.
     */
    exclusiveUnionable: function( interval ) {
      return ( !this.a || !interval.b || this.a.compare( interval.b ) === -1 ) &&
             ( !this.b || !interval.a || this.b.compare( interval.a ) === 1 );
    },
    
    union: function( interval ) {
      return new TrailInterval(
        this.a.compare( interval.a ) === -1 ? this.a : interval.a,
        this.b.compare( interval.b ) ===  1 ? this.b : interval.b
      );
    }
  };
  
  return TrailInterval;
} );


