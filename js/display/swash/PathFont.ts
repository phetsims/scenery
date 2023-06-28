// Copyright 2023, University of Colorado Boulder

/**
 * Loading utilities for the async nature of using swash
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import swash, { SwashFont } from './swash.js';
import swash_wasm from './swash_wasm.js';
import { base64ToU8, scenery } from '../../imports.js';
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import { Shape } from '../../../../kite/js/imports.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';

const loadPromise = swash( base64ToU8( swash_wasm ) );

// Record whether it's loaded, so we can skip the asyncLoader lock if so
let loaded = false;
loadPromise.then( () => {
  loaded = true;
} ).catch( ( err: IntentionalAny ) => {
  throw err;
} );

export default class PathFont {

  // Glyphs come out in units that are typically 1000 or 2048 per EM, so if we're scaling to a font size of
  // 16, we need to scale by 16 / unitsPerEM.
  public unitsPerEM = 0;

  private swashFont: SwashFont | null = null;
  private readonly glyphCache: Map<number, Shape | null> = new Map<number, Shape | null>();

  public constructor( dataString: string ) {
    if ( loaded ) {
      this.initialize( dataString );
    }
    else {
      const lock = asyncLoader.createLock( 'font load' );
      loadPromise.then( () => {

        this.initialize( dataString );

        lock();
      } ).catch( ( err: IntentionalAny ) => { throw err; } );
    }
  }

  public initialize( dataString: string ): void {
    this.swashFont = new SwashFont( base64ToU8( dataString ) );
    this.unitsPerEM = this.swashFont.get_units_per_em();
  }

  // If ANY glyph isn't included, we bail out, and null is returned
  public shapeText( str: string, ltr: boolean ): PathGlyph[] | null {
    if ( !this.swashFont ) {
      throw new Error( 'Font not loaded yet' );
    }

    let success = true;
    const pathGlyphs = JSON.parse( this.swashFont.shape_text( str, ltr ) ).map( ( glyph: { x: number; y: number; id: number; adv: number } ) => {
      const shape = this.getGlyph( glyph.id );
      if ( shape ) {
        return new PathGlyph( glyph.id, shape, glyph.x, glyph.y, glyph.adv );
      }
      else {
        success = false;
        return null;
      }
    } );

    return success ? pathGlyphs : null;
  }

  public getGlyph( id: number ): Shape | null {
    if ( !this.swashFont ) {
      throw new Error( 'Font not loaded yet' );
    }

    if ( !this.glyphCache.has( id ) ) {
      // No embolden for now, if we're trying to use fonts directly
      const glyphResult = this.swashFont.get_glyph( id, 0, 0 );
      this.glyphCache.set( id, glyphResult === 'MISSING' ? null : new Shape( glyphResult ).makeImmutable() );
    }
    return this.glyphCache.get( id )!;
  }
}

scenery.register( 'PathFont', PathFont );

export class PathGlyph {
  public constructor(
    public id: number,
    public shape: Shape,
    public x: number,
    public y: number,
    public advance: number
  ) {}
}
