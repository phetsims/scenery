// Copyright 2023, University of Colorado Boulder

/**
 * Representation of a winding map for a face (or an edge as a delta)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';
import { RenderPath } from '../render-program/RenderProgram.js';

export default class WindingMap {
  public constructor( public readonly map: Map<RenderPath, number> = new Map() ) {}

  public getWindingNumber( renderPath: RenderPath ): number {
    return this.map.get( renderPath ) || 0;
  }

  public addWindingNumber( renderPath: RenderPath, amount: number ): void {
    const current = this.getWindingNumber( renderPath );
    this.map.set( renderPath, current + amount );
  }

  public addWindingMap( windingMap: WindingMap ): void {
    for ( const [ renderPath, winding ] of windingMap.map ) {
      this.addWindingNumber( renderPath, winding );
    }
  }

  public equals( windingMap: WindingMap ): boolean {
    if ( this.map.size !== windingMap.map.size ) {
      return false;
    }
    for ( const [ renderPath, winding ] of this.map ) {
      if ( winding !== windingMap.getWindingNumber( renderPath ) ) {
        return false;
      }
    }
    return true;
  }
}

scenery.register( 'WindingMap', WindingMap );
