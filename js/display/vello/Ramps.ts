// Copyright 2023, University of Colorado Boulder

/**
 * Handles caching of ramps (for gradients) and their texture representation.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { lerpRGBA8, scenery, premultiplyRGBA8, VelloColorStop, VelloRampPatch } from '../../imports.js';

const NUM_RAMP_SAMPLES = 512;
const STARTING_RAMPS = 32;

export default class Ramps {

  private arrayBuffer: ArrayBuffer;
  private arrayView: DataView;
  public width: number;
  public height: number;
  public texture: GPUTexture | null = null;
  public textureView: GPUTextureView | null = null;

  public ramps: RampEntry[] = []; // by index
  public rampMap: Map<string, RampEntry> = new Map<string, RampEntry>(); // RampEntry.getMapKey() => RampEntry
  public dirty = false; // initially because it's empty! -- done after replaceTexture() so we don't start dirty
  public generation = 0;

  public constructor( public readonly device: GPUDevice ) {

    // NOTE: we can increase the size in the future
    // TODO: test the increase in size, and other ramp characteristics
    this.arrayBuffer = new ArrayBuffer( NUM_RAMP_SAMPLES * STARTING_RAMPS * 4 );
    this.arrayView = new DataView( this.arrayBuffer );
    this.width = NUM_RAMP_SAMPLES;
    this.height = STARTING_RAMPS;

    this.replaceTexture();
  }

  public updatePatches( patches: VelloRampPatch[] ): void {
    const generation = this.generation++;

    patches.forEach( ( patch, i ) => {
      const mapKey = RampEntry.getMapKey( patch.stops );
      let rampEntry = this.rampMap.get( mapKey );
      if ( rampEntry ) {
        rampEntry.generation = generation;
        patch.id = rampEntry.index;
      }
      else {
        let newIndex;

        if ( this.ramps.length < this.height ) {
          newIndex = this.ramps.length;
        }
        else {
          const oldEntry = _.find( this.ramps, entry => entry.generation < generation - 1 );
          if ( oldEntry ) {
            this.rampMap.delete( oldEntry.getMapKey() );
            newIndex = oldEntry.index;
          }
          else {
            // Increase size!
            this.height *= 2;
            const newArrayBuffer = new ArrayBuffer( this.arrayBuffer.byteLength * 2 );
            const newArrayView = new DataView( newArrayBuffer );
            new Uint8Array( newArrayBuffer ).set( new Uint8Array( this.arrayBuffer ) ); // data copy (what is there)
            this.arrayBuffer = newArrayBuffer;
            this.arrayView = newArrayView;
            this.replaceTexture();

            newIndex = this.ramps.length;
          }
        }

        rampEntry = new RampEntry( patch.stops, newIndex, generation );
        this.ramps[ newIndex ] = rampEntry;
        this.rampMap.set( mapKey, rampEntry );
        patch.id = newIndex;
        this.writeRamp( rampEntry.index, patch.stops );
      }
    } );
  }

  private replaceTexture(): void {
    this.texture && this.texture.destroy();

    this.texture = this.device.createTexture( {
      label: 'ramps texture',
      size: {
        width: this.width || 1,
        height: this.height || 1,
        depthOrArrayLayers: 1
      },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    } );

    this.textureView = this.texture.createView( {
      label: 'ramps texture view',
      format: 'rgba8unorm',
      dimension: '2d'
    } );

    this.dirty = true;
  }

  public updateTexture(): void {
    if ( this.dirty ) {
      this.dirty = false;

      assert && assert( this.texture );

      this.device.queue.writeTexture( {
        texture: this.texture!
      }, this.arrayBuffer, {
        offset: 0,
        bytesPerRow: this.width * 4
      }, {
        width: this.width,
        height: this.height,
        depthOrArrayLayers: 1
      } );
    }
  }

  private writeRamp( index: number, colorStops: VelloColorStop[] ): void {
    this.dirty = true;

    const offset = index * NUM_RAMP_SAMPLES * 4;

    let last_u = 0.0;
    let last_c = colorStops[ 0 ].color;
    let this_u = last_u;
    let this_c = last_c;
    let j = 0;

    for ( let i = 0; i < NUM_RAMP_SAMPLES; i++ ) {
      const u = i / ( NUM_RAMP_SAMPLES - 1 );
      while ( u > this_u ) {
        last_u = this_u;
        last_c = this_c;
        const colorStop = colorStops[ j ];
        if ( colorStop ) {
          this_u = colorStop.offset;
          this_c = colorStop.color;
          j++;
        }
        else {
          break;
        }
      }
      const du = this_u - last_u;
      const u32 = premultiplyRGBA8( du < 1e-9 ? this_c : lerpRGBA8( last_c, this_c, ( u - last_u ) / du ) );
      this.arrayView.setUint32( offset + i * 4, u32, false );
    }
  }

  public dispose(): void {
    this.texture && this.texture.destroy();
  }
}

scenery.register( 'Ramps', Ramps );

class RampEntry {
  // TODO: make poolable?
  public constructor(
    public readonly colorStops: VelloColorStop[],
    public readonly index: number,
    public generation: number
  ) {}

  public getMapKey(): string {
    return RampEntry.getMapKey( this.colorStops );
  }

  public static getMapKey( colorStops: VelloColorStop[] ): string {
    return colorStops.map( colorStop => `${colorStop.offset}-${colorStop.color}` ).join( ',' );
  }
}