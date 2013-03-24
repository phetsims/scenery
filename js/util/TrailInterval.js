// Copyright 2002-2012, University of Colorado

/**
 * An interval between two Trails.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Transform3 = require( 'DOT/Transform3' );
  
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
    
    disjointFrom: function( interval, excludeEndTrails ) {
      // TODO: handle null (like infinity!)
      var c = this.a.compare( interval.b );
      var d = this.b.compare( interval.a );
      
      throw new Error( 'needs to handle null case' );
      
      if ( c === d && c !== 0 ) {
        return true;
      }
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


