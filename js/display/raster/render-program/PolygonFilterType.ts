// Copyright 2023, University of Colorado Boulder

/**
 * Controls how polygons get filtered when output
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

enum PolygonFilterType {
  Box = 0,
  Bilinear = 1,
  MitchellNetravali = 2
}

export default PolygonFilterType;

export const getPolygonFilterExtraPixels = ( filterType: PolygonFilterType ): number => {
  if ( filterType === PolygonFilterType.Box ) {
    return 0;
  }
  else if ( filterType === PolygonFilterType.Bilinear ) {
    return 1;
  }
  else if ( filterType === PolygonFilterType.MitchellNetravali ) {
    return 3;
  }
  else {
    throw new Error( `Unknown PolygonFilterType: ${filterType}` );
  }
};

export const getPolygonFilterGridOffset = ( filterType: PolygonFilterType ): number => {
  if ( filterType === PolygonFilterType.Box ) {
    return 0;
  }
  else if ( filterType === PolygonFilterType.Bilinear ) {
    return -0.5;
  }
  else if ( filterType === PolygonFilterType.MitchellNetravali ) {
    return -1.5;
  }
  else {
    throw new Error( `Unknown PolygonFilterType: ${filterType}` );
  }
};

export const getPolygonFilterMinExpand = ( filterType: PolygonFilterType ): number => {
  if ( filterType === PolygonFilterType.Box ) {
    return 0;
  }
  else if ( filterType === PolygonFilterType.Bilinear ) {
    return 1;
  }
  else if ( filterType === PolygonFilterType.MitchellNetravali ) {
    return 2;
  }
  else {
    throw new Error( `Unknown PolygonFilterType: ${filterType}` );
  }
};

export const getPolygonFilterMaxExpand = ( filterType: PolygonFilterType ): number => {
  if ( filterType === PolygonFilterType.Box ) {
    return 1;
  }
  else if ( filterType === PolygonFilterType.Bilinear ) {
    return 1;
  }
  else if ( filterType === PolygonFilterType.MitchellNetravali ) {
    return 2;
  }
  else {
    throw new Error( `Unknown PolygonFilterType: ${filterType}` );
  }
};

scenery.register( 'PolygonFilterType', PolygonFilterType );
