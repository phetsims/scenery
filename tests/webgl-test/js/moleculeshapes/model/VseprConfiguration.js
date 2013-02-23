// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

  var model = phet.moleculeshapes.model;

  phet.moleculeshapes.model.VseprConfiguration = function ( x, e ) {
    this.x = x;
    this.e = e;

    this.geometry = model.GeometryConfiguration.getConfiguration( x + e ); // undefined?
    this.bondedUnitVectors = [];
    this.lonePairUnitVectors = [];
    for ( var i = 0; i < x + e; i++ ) {
      if ( i < e ) {
        // fill up the lone pair unit vectors first
        this.lonePairUnitVectors.push( this.geometry.unitVectors[i] );
      }
      else {
        this.bondedUnitVectors.push( this.geometry.unitVectors[i] );
      }
    }

    // figure out what the name is
    if ( x == 0 ) {
      this.name = Strings.SHAPE__EMPTY;
    }
    else if ( x == 1 ) {
      this.name = Strings.SHAPE__DIATOMIC;
    }
    else if ( x == 2 ) {
      if ( e == 0 || e == 3 || e == 4 ) {
        this.name = Strings.SHAPE__LINEAR;
      }
      else if ( e == 1 || e == 2 ) {
        this.name = Strings.SHAPE__BENT;
      }
      else {
        throw new Error( "invalid x: " + x + ", e: " + e );
      }
    }
    else if ( x == 3 ) {
      if ( e == 0 ) {
        this.name = Strings.SHAPE__TRIGONAL_PLANAR;
      }
      else if ( e == 1 ) {
        this.name = Strings.SHAPE__TRIGONAL_PYRAMIDAL;
      }
      else if ( e == 2 || e == 3 ) {
        this.name = Strings.SHAPE__T_SHAPED;
      }
      else {
        throw new Error( "invalid x: " + x + ", e: " + e );
      }
    }
    else if ( x == 4 ) {
      if ( e == 0 ) {
        this.name = Strings.SHAPE__TETRAHEDRAL;
      }
      else if ( e == 1 ) {
        this.name = Strings.SHAPE__SEESAW;
      }
      else if ( e == 2 ) {
        this.name = Strings.SHAPE__SQUARE_PLANAR;
      }
      else {
        throw new Error( "invalid x: " + x + ", e: " + e );
      }
    }
    else if ( x == 5 ) {
      if ( e == 0 ) {
        this.name = Strings.SHAPE__TRIGONAL_BIPYRAMIDAL;
      }
      else if ( e == 1 ) {
        this.name = Strings.SHAPE__SQUARE_PYRAMIDAL;
      }
      else {
        throw new Error( "invalid x: " + x + ", e: " + e );
      }
    }
    else if ( x == 6 ) {
      if ( e == 0 ) {
        this.name = Strings.SHAPE__OCTAHEDRAL;
      }
      else {
        throw new Error( "invalid x: " + x + ", e: " + e );
      }
    }
    else {
      this.name = null;
    }
  };

  var VseprConfiguration = phet.moleculeshapes.model.VseprConfiguration;

  var Strings = phet.moleculeshapes.strings;
  var Vector3 = dot.Vector3;

  VseprConfiguration.prototype = {
    constructor: VseprConfiguration,

    getAllUnitVectors: function () {
      return this.geometry.unitVectors;
    },

    getIdealBondUnitVectors: function () {
      var result = [];
      for ( var i = e; i < x + e; i++ ) {
        result.push( this.geometry.unitVectors.get( i ) );
      }
      return result;
    },

    // for finding ideal rotations including matching for "bond-vs-bond" and "lone pair-vs-lone pair"
    getIdealGroupRotationToPositions: function ( groups ) {
      phet.assert( ( x + e ) == groups.length );

      // done currently only when the molecule is rebuilt, so we don't try to pass a lastPermutation in (not helpful)
      return model.AttractorModel.findClosestMatchingConfiguration( model.AttractorModel.getOrientationsFromOrigin( groups ), this.geometry.unitVectors, model.LocalShape.vseprPermutations( groups ) );
    },

    // for finding ideal rotations exclusively using the "bonded" portions
    getIdealBondRotationToPositions: function ( groups ) {
      // ideal vectors excluding lone pairs (just for the bonds)
      phet.assert( ( x ) == groups.length );
      var idealModelBondVectors = this.getIdealBondUnitVectors();

      // currently only called when a real molecule is built, so we don't try to pass a lastPermutation in (not helpful)
      return model.AttractorModel.findClosestMatchingConfiguration( model.AttractorModel.getOrientationsFromOrigin( groups ), idealModelBondVectors, dot.Permutation.permutations( idealModelBondVectors.length ) );
    },

    equals: function ( other ) {
      return this.x == other.x && this.e == other.e;
    }
  };
})();
